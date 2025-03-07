import logging
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union

import torch
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from sklearn.metrics import roc_auc_score
from torch.nn import MultiheadAttention, LayerNorm
from torchvision.models.vision_transformer import MLP

from architectures.early_exits.pbee import PBEE
from architectures.vit import MLP as CustomMLP
from architectures import VAR
from utils import flop_count, get_module_by_name, remove_hooks, find_module_names, add_save_activations_hook
from architectures.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
#import utils
import numpy as np
import dist
import pickle
import os

#from trainer import VARTrainer


def test_classification(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        tc, 
                        batches: int = 0) -> Tuple[float, float]:

    if any(isinstance(submodule, VAR) for submodule in model.modules()):
        # model = accelerator.unwrap_model(model)
        L_mean, L_tail, acc_mean, acc_tail, total, elapsed_time = tc.trainer.eval_ep(tc.val_loader)
        mean_loss = L_mean
        accuracy = acc_mean
    else:
        criterion = criterion_class(reduction='sum')
        model.eval()
        with torch.inference_mode():
            running_loss = 0.0
            correct, total = 0, 0
            for batch, (X, y) in enumerate(data_loader):
                y_pred = model(X)
                y_pred, y = accelerator.gather_for_metrics((y_pred, y))
                y_pred_max = y_pred.argmax(dim=-1)
                if len(y_pred.shape) == 3:
                    # For CE loss on sequences
                    y_pred = y_pred.transpose(1, 2)
                loss = criterion(y_pred, y)
                running_loss += loss.item()
                correct += (y_pred_max == y).sum().item()
                # Again account for multi-dimensional targets
                total += y.numel()
                if batches > 0 and batch == batches - 1:
                    break
        mean_loss = running_loss / total
        accuracy = correct / total
    # loss, acc
    return mean_loss, accuracy


def get_preds(accelerator: Accelerator,
              model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    return batch_outputs, batch_labels


def get_preds_earlyexiting(accelerator: Accelerator,
                           model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           batches: int = 0):
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    batch_outputs = []
    batch_labels = []
    with torch.inference_mode():
        unwrapped_model.all_mode()
        for batch, (X, y) in enumerate(data_loader):
            output = model(X)
            output, y = accelerator.gather_for_metrics((output, y))
            y_preds = [y_pred.detach().cpu() for y_pred in output]
            batch_outputs.append(y_preds)
            batch_labels.append(y.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_head_preds = []
    for i in range(unwrapped_model.number_of_heads):
        head_outputs = torch.cat([batch_output[i] for batch_output in batch_outputs])
        batch_head_preds.append(head_outputs)
    batch_labels = torch.cat(batch_labels)
    return batch_head_preds, batch_labels


def get_preds_moe(accelerator: Accelerator,
                  model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    batch_gating_data = defaultdict(list)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            output, gating_data = model(X, return_gating_data=True)
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            output, y, gating_data = accelerator.gather_for_metrics((output, y, gating_data))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            for k, v in gating_data.items():
                batch_gating_data[k].append(v.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    for k in batch_gating_data.keys():
        batch_gating_data[k] = torch.cat(batch_gating_data[k])
    return batch_outputs, batch_labels, batch_gating_data


def get_preds_avit(accelerator: Accelerator,
                   model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   batches: int = 0):
    model.eval()
    batch_outputs = []
    batch_labels = []
    batch_token_counts = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            output, token_counts = model(X, return_counts=True)
            output, y, token_counts = accelerator.gather_for_metrics((output, y, token_counts))
            batch_outputs.append(output.detach().cpu())
            batch_labels.append(y.detach().cpu())
            batch_token_counts.append(token_counts.detach().cpu())
            if batches > 0 and batch == batches - 1:
                break
    batch_outputs = torch.cat(batch_outputs)
    batch_labels = torch.cat(batch_labels)
    batch_token_counts = torch.cat(batch_token_counts)
    return batch_outputs, batch_labels, batch_token_counts


def online_evaluate_moe(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        cost_without_experts,
                        token_expert_costs,
                        batches: int = 0,
                        return_counts: bool = False):
    criterion = criterion_class(reduction='sum')
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    total_average_flops = cost_without_experts
    executed_expert_tokens = {name: 0 for name in token_expert_costs.keys()}
    total_expert_tokens = {name: 0 for name in token_expert_costs.keys()}
    expert_average_costs = {}
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            y_pred, gating_data = model(X, return_gating_data=True)
            # each element of gating_data_list is a tuple
            # because different MoEs classes can return more than simply the gating network's final outputs
            # we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y, gating_data = accelerator.gather_for_metrics((y_pred, y, gating_data))
            y_pred_max = y_pred.argmax(dim=-1)
            if len(y_pred.shape) == 3:
                # For CE loss on sequences
                y_pred = y_pred.transpose(1, 2)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            total += y.numel()  # use numel since targets can be batches of sequences
            for moe_name in token_expert_costs.keys():
                executed_expert_tokens[moe_name] += (gating_data[moe_name] > 0.0).long().sum().item()
                total_expert_tokens[moe_name] += gating_data[moe_name].numel()
            if batches > 0 and batch == batches - 1:
                break
    for moe_name, token_expert_cost in token_expert_costs.items():
        expert_average_cost = executed_expert_tokens[moe_name] * token_expert_cost / total
        logging.info(f'Averaged FLOPs for MoE {moe_name}: {expert_average_cost}')
        expert_average_costs[moe_name] = expert_average_cost
        total_average_flops += expert_average_cost
    # loss, acc
    if return_counts:
        return running_loss / total, correct / total, total_average_flops, expert_average_costs, \
            executed_expert_tokens, total_expert_tokens
    else:
        return running_loss / total, correct / total, total_average_flops, expert_average_costs



def evaluate_model_throughput(model: torch.nn.Module,
                              data_loader: torch.utils.data.DataLoader,
                              criterion_class: torch.nn.Module,
                              batches: int = 0,
                              device='cpu',
                              warmup_rounds: int = 3):
    criterion = criterion_class(reduction='sum')
    model.eval()
    running_loss = torch.tensor(0.0, dtype=torch.double, device=device)
    correct, total = torch.tensor(0.0, dtype=torch.long, device=device), 0
    #
    with torch.inference_mode():
        # warmup
        logging.info(f'Warming up the model for throughput measurements...')
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _y_pred = model(X)
            if batch >= warmup_rounds:
                break
        if 'cuda' in device:
            torch.cuda.synchronize()
        logging.info(f'Model warmed-up, starting measurements...')
        # torch.cuda.set_sync_debug_mode("error")
        start = time.monotonic()
        for batch, (X, y) in enumerate(data_loader):
            total += y.size(0)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(X)
            y_pred_max = y_pred.argmax(dim=1)
            loss = criterion(y_pred, y)
            running_loss += loss.detach()
            correct += (y_pred_max == y).sum().detach()
            if batches > 0 and batch == batches - 1:
                break
        # torch.cuda.set_sync_debug_mode(0)
        if 'cuda' in device:
            torch.cuda.synchronize()
        stop = time.monotonic()
        duration = stop - start
    #
    dataset_size = len(data_loader.dataset)
    return running_loss / total, correct / total, total / duration, duration, dataset_size


def test_earlyexiting_classification(accelerator: Accelerator,
                                     model: torch.nn.Module,
                                     data_loader: torch.utils.data.DataLoader,
                                     criterion_class: torch.nn.Module,
                                     batches: int = 0):
    criterion = criterion_class(reduction='mean')
    head_preds, ys = get_preds_earlyexiting(accelerator, model, data_loader, batches)
    head_losses = [criterion(preds, ys) for preds in head_preds]
    head_accuracies = [(preds.argmax(dim=1) == ys).sum().item() / ys.size(0) for preds in head_preds]
    return head_losses, head_accuracies


def average_earlyexiting_flops(head_costs: List, head_exit_counts: torch.Tensor):
    assert len(head_costs) == head_exit_counts.size(0), f'{head_costs=}\n{head_exit_counts=}'
    total_cost = 0.0
    for h_i, h_c in enumerate(head_costs):
        total_cost += h_c * head_exit_counts[h_i].item()
    average_cost = total_cost / head_exit_counts.sum().item()
    return average_cost


def average_avit_flops(constant_cost, mha_sequence_costs, mlp_token_cost, token_counts):
    mha_sequence_costs = torch.tensor(mha_sequence_costs, dtype=torch.long)
    # token counts contains the number of layers (i.e. entire blocks)
    # executed by the model for each token
    total_average_flops = constant_cost
    for layer_i in range(token_counts.max().item()):
        current_sequence_lengths = (token_counts > layer_i).to(torch.long).sum(dim=1).squeeze(-1)
        mha_average_cost = torch.gather(mha_sequence_costs, dim=0,
                                        index=current_sequence_lengths).sum().item() / token_counts.size(0)
        mlp_average_cost = mlp_token_cost * (token_counts > layer_i).to(torch.long).sum().item() / token_counts.size(0)
        total_average_flops += mha_average_cost + mlp_average_cost
    logging.info(f'Total average model cost: {total_average_flops}')
    return total_average_flops


def evaluate_earlyexiting_classification(model: torch.nn.Module,
                                         head_preds: List[torch.Tensor],
                                         labels: torch.Tensor,
                                         head_costs: List[FlopCountAnalysis],
                                         eval_thresholds: int) -> Dict:
    head_accuracies = []
    for i, head_pred in enumerate(head_preds):
        head_accuracy = (head_pred.argmax(dim=1) == labels).sum().item() / labels.size(0)
        head_accuracies.append(head_accuracy)
    head_flops = [head_cost.total() for head_cost in head_costs]
    thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
    threshold_accuracies = []
    threshold_flops = []
    # separate path for evaluating PBEE
    if isinstance(model, PBEE):
        patience_thresholds = torch.arange(0, len(head_preds), device=labels.device)
        for patience_threshold in patience_thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_preds[0])
            # start from second head, set patience to one after first head
            prev_answers = torch.zeros_like(labels) - 1
            patience = torch.zeros_like(prev_answers)
            for i, head_pred in enumerate(head_preds):
                patience = torch.where(head_pred.argmax(-1) == prev_answers, patience + 1, 1)
                unresolved_mask = exit_at == -1
                exit_mask = (patience > patience_threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_pred[exit_mask]
                prev_answers = head_pred.argmax(-1)
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_preds[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_preds) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            exits_bincounted = exit_at.bincount(minlength=len(head_preds))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_flops.append(threshold_cost)
    else:
        head_probs = [preds.softmax(dim=-1) for preds in head_preds]
        thresholds = torch.linspace(0.0, 1.0, steps=eval_thresholds, device=labels.device)
        for threshold in thresholds:
            exit_at = torch.zeros_like(labels) - 1
            outputs = torch.zeros_like(head_probs[0])
            for i, head_prob in enumerate(head_probs):
                head_confidences, _ = head_prob.max(dim=-1)
                unresolved_mask = exit_at == -1
                exit_mask = (head_confidences > threshold) & unresolved_mask
                exit_at[exit_mask] = i
                outputs[exit_mask] = head_prob[exit_mask]
            unresolved_mask = exit_at == -1
            outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
            exit_at[unresolved_mask] = len(head_probs) - 1
            threshold_accuracy = ((outputs.argmax(dim=-1) == labels).sum() / labels.size(0)).item()
            exits_bincounted = exit_at.bincount(minlength=len(head_probs))
            threshold_cost = average_earlyexiting_flops(head_flops, exits_bincounted)
            threshold_accuracies.append(threshold_accuracy)
            threshold_flops.append(threshold_cost)
    results = {'head_scores': head_accuracies, 'head_flops': head_flops, 'thresholds': thresholds,
               'threshold_scores': threshold_accuracies, 'threshold_flops': threshold_flops}
    return results


def evaluate_classification(preds: torch.Tensor, labels: torch.Tensor, criterion_class: torch.nn.Module):
    criterion = criterion_class(reduction='mean')
    preds_max = preds.argmax(dim=1)
    loss = criterion(preds, labels).item()
    accuracy = (preds_max == labels).double().mean().item()
    return loss, accuracy


def ks_calibration_error(probs, labels):
    '''https://arxiv.org/abs/2006.12800'''
    assert probs.dim() == 2, f'{probs.size()=}'
    num_classes = probs.size(-1)
    labels_oh = torch.nn.functional.one_hot(labels, num_classes)
    num_samples = probs.size(0)
    ks_errors = [0.0] * num_classes
    for k in range(num_classes):
        class_probs = probs[..., k]
        class_labels = labels_oh[..., k]
        sorted_probs, indices = class_probs.sort()
        h_tilde = torch.cumsum(class_labels[indices] / num_samples, dim=0)
        h = torch.cumsum(sorted_probs / num_samples, dim=0)
        ks_errors[k] += (h - h_tilde).abs().max().item()
    # TODO is averaging appropriate?
    ks_error = sum(ks_errors) / num_classes
    return ks_error, ks_errors


def evaluate_calibration(preds: torch.Tensor,
                         labels: torch.Tensor) -> Dict:
    probs = preds.softmax(dim=-1)
    # ignores per-class calibration scores, takes the average
    calibration_score, _ = ks_calibration_error(probs, labels)
    results = {'final_score': calibration_score}
    return results


# TODO possibly generalize this code and merge it with accuracy and ood
def evaluate_earlyexiting_calibration(head_preds: List[torch.Tensor],
                                      labels: torch.Tensor,
                                      head_costs: List[int],
                                      thresholds: torch.Tensor) -> Dict:
    head_probs = [preds.softmax(dim=-1) for preds in head_preds]
    head_calibration_scores = []
    for i, head_prob in enumerate(head_probs):
        head_calibration_score, _ = ks_calibration_error(head_prob, labels)
        head_calibration_scores.append(head_calibration_score)
    threshold_calibration_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(labels) - 1
        outputs = torch.zeros_like(head_probs[0])
        for i, head_prob in enumerate(head_probs):
            head_confidences, _ = head_prob.max(dim=-1)
            unresolved_mask = exit_at == -1
            exit_mask = (head_confidences > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_prob[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_probs[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_probs) - 1
        threshold_calibration_score, _ = ks_calibration_error(outputs, labels)
        exits_bincounted = exit_at.bincount(minlength=len(head_probs))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_calibration_scores.append(threshold_calibration_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_calibration_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_calibration_scores, 'threshold_flops': threshold_flops}
    return results


def evaluate_ood_detection(id_preds: List[torch.Tensor],
                           ood_preds: torch.Tensor) -> Dict:
    id_confidences = id_preds.softmax(dim=-1).max(dim=-1)[0]
    ood_confidences = ood_preds.softmax(dim=-1).max(dim=-1)[0]
    confidences = torch.cat([id_confidences, ood_confidences])
    ood_labels = torch.cat([torch.ones_like(id_confidences), torch.zeros_like(ood_confidences)])
    ood_score = roc_auc_score(ood_labels.cpu().numpy(), confidences.cpu().numpy())
    assert 0.0 <= ood_score <= 1.0, f'AUROC: {ood_score}'
    results = {'final_score': ood_score}
    return results


def evaluate_earlyexiting_ood_detection(head_id_preds: List[torch.Tensor],
                                        head_ood_preds: List[torch.Tensor],
                                        head_costs: List[int],
                                        thresholds: torch.Tensor) -> Dict:
    # TODO this assumes the head costs are the same for the OOD dataset - add support for different costs
    head_id_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_id_preds]
    head_ood_confidences = [preds.softmax(dim=-1).max(dim=-1)[0] for preds in head_ood_preds]
    head_confidences = [torch.cat([id_confidences, ood_confidences]) for id_confidences, ood_confidences in
                        zip(head_id_confidences, head_ood_confidences)]
    ood_labels = torch.cat([torch.ones_like(head_id_confidences[0], dtype=torch.int),
                            torch.zeros_like(head_ood_confidences[0], dtype=torch.int)])
    head_ood_scores = []
    for i, head_confs in enumerate(head_confidences):
        head_ood_score = roc_auc_score(ood_labels.cpu().numpy(), head_confs.cpu().numpy())
        head_ood_scores.append(head_ood_score)
    threshold_ood_scores = []
    threshold_flops = []
    for threshold in thresholds:
        exit_at = torch.zeros_like(ood_labels) - 1
        outputs = torch.zeros_like(head_confidences[0])
        for i, head_confs in enumerate(head_confidences):
            unresolved_mask = exit_at == -1
            exit_mask = (head_confs > threshold) & unresolved_mask
            exit_at[exit_mask] = i
            outputs[exit_mask] = head_confs[exit_mask]
        unresolved_mask = exit_at == -1
        outputs[unresolved_mask] = head_confidences[-1][unresolved_mask]
        exit_at[unresolved_mask] = len(head_confidences) - 1
        threshold_ood_detection_score = roc_auc_score(ood_labels.cpu().numpy(), outputs.cpu().numpy())
        exits_bincounted = exit_at.bincount(minlength=len(head_confidences))
        threshold_cost = average_earlyexiting_flops(head_costs, exits_bincounted)
        threshold_ood_scores.append(threshold_ood_detection_score)
        threshold_flops.append(threshold_cost)
    results = {'head_scores': head_ood_scores, 'head_flops': head_costs, 'thresholds': thresholds,
               'threshold_scores': threshold_ood_scores, 'threshold_flops': threshold_flops}
    return results


def benchmark_with_sample(model: torch.nn.Module,
                          sample: torch.tensor) -> Tuple[FlopCountAnalysis, Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    with torch.inference_mode():
        model_costs = flop_count(model, (sample))
        param_count = parameter_count(model)
    #logging.info(f'Ops by operator:\n{model_costs.by_operator()}')
    #logging.info(f'Ops by module:\n{flop_count_table(model_costs, max_depth=7)}') # Jort to Add again right now this is long
    logging.info(f'Total ops: {model_costs.total()}')
    unsupported = model_costs.unsupported_ops()
    # if len(unsupported) > 0:
    #     logging.warning("Unsupported ops: " + ", ".join(f"{k} (occurrences: {v})" for k, v in unsupported.items()))

    uncalled = model_costs.uncalled_modules()
    # if len(uncalled) > 0:
    #     logging.warning(f'Uncalled modules: {", ".join(str(m) for m in uncalled)}')

    return model_costs, param_count


def benchmark(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader, tc) -> Tuple[FlopCountAnalysis, Dict]:
    X, y = next(iter(data_loader))
    
    with torch.no_grad():
        B, V = y.shape[0], tc.model_vae.vocab_size
        X = X.to(dist.get_device(), non_blocking=True)
        label_B = y.to(dist.get_device(), non_blocking=True)
        gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X) # This does not return None
        x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
    
    sample = (label_B, x_BLCv_wo_first_l)

    return benchmark_with_sample(model, sample)





def benchmark_earlyexiting(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader) \
        -> Tuple[List[FlopCountAnalysis], Dict]:
    model.eval()
    # workaround for the missing implementation of 'aten::_native_multi_head_attention' flop counter
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.train()
    #
    X, _ = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[0].unsqueeze(0) for k, v in X.items()}
    else:
        sample = X[0].unsqueeze(0)
    with torch.inference_mode():
        param_count = parameter_count(model)
        head_costs = []
        for head_i in range(model.number_of_heads):
            model.select_head(head_i)
            head_costs.append(flop_count(model, (sample,)))
            logging.info(f'Ops for head {head_i}: {head_costs[head_i].total()}')
    unsupported = head_costs[-1].unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f'Unsupported op: {k} (occurrences: {v})')
    uncalled = head_costs[-1].uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f'Uncalled module: {m}')
    return head_costs, param_count

import torch
from typing import Tuple, List
import matplotlib.pyplot as plt


def score_moe(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              tc, tau: float) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the MoE model using TAU-based filtering and visualize predictions.
    
    This function computes:
      - `L_mean`: Mean loss over all tokens.
      - `L_tail`: Mean loss over only the final `tc.trainer.last_l` tokens.
      - `acc_mean`: Token-level accuracy over the entire sequence.
      - `acc_tail`: Token-level accuracy over the last `tc.trainer.last_l` tokens.
      - `average_experts_per_token`: The number of experts used per token on average.
    
    Additionally, it visualizes **the last batch of predictions vs. ground truth** images.
    """
    model.eval()
    total_L_mean = 0.0
    total_L_tail = 0.0
    total_acc_mean = 0.0
    total_acc_tail = 0.0
    total_experts_selected = 0.0
    total_tokens = 0
    total_samples = 0

    device = next(model.parameters()).device

    # Identify MoE modules
    moe_modules = [m for m in model.modules() if hasattr(m, 'gate') and hasattr(m, 'router')]
    if not moe_modules:
        raise ValueError("No MoE modules (with gate and router) were found in the model.")

    # Save original gate functions
    original_gates = {m: m.gate for m in moe_modules}

    # Define a custom gate function for the first pass that captures the input.
    def capturing_gate(x_in, m):
        # Store the tokenized input (the representation fed into the MoE module)
        m.last_router_input = x_in
        # Return a tensor of ones to force using all experts.
        return torch.ones(x_in.size(0), x_in.size(1), m.num_experts,
                          device=x_in.device, dtype=x_in.dtype)

    total_batches = 3  # Limit to 5 batches for evaluation
    last_inp_B3HW = None
    last_logits_BLV = None

    with torch.no_grad():
        for batch in data_loader:
            if total_batches == 0:
                break
            total_batches -= 1

            inp_B3HW, label_B = batch
            B = inp_B3HW.size(0)

            # Convert images to token sequences
            inp_B3HW = inp_B3HW.to(device, non_blocking=True)
            label_B  = label_B.to(device, non_blocking=True)

            gt_idx_Bl: List[torch.Tensor] = tc.model_vae.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1).to(device)  # Shape: (B, L)
            x_BLCv_wo_first_l: torch.Tensor = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            L = gt_BL.shape[1]  # Sequence length
            V = tc.model_vae.vocab_size  # Vocabulary size

            # --- First Pass: Capture the token representation while forcing all experts ---
            for moe in moe_modules:
                moe.gate = lambda x_in, m=moe: capturing_gate(x_in, m)
            
            model(label_B, x_BLCv_wo_first_l)  # Captures `last_router_input` in each MoE module.

            # --- Build TAU-based Masks for Each MoE Module ---
            for moe in moe_modules:
                if not hasattr(moe, "last_router_input"):
                    raise ValueError("MoE module did not capture router input in the first pass.")
                router_input = moe.last_router_input  # Expected shape: (B, T, D)
                
                # Check input dimension matches router's expected input.
                expected_dim = moe.router.layers[0].in_features
                if router_input.size(-1) != expected_dim:
                    if hasattr(moe, "router_proj"):
                        router_input = moe.router_proj(router_input)
                    else:
                        raise ValueError(
                            f"Router input dimension mismatch: got {router_input.size(-1)}, expected {expected_dim}."
                        )

                # Compute predicted expert norms and TAU-based mask
                predicted_expert_norms = moe.router(router_input)  # Shape: (B, T, num_experts)
                max_norms, _ = predicted_expert_norms.max(dim=-1, keepdim=True)
                thresholds = max_norms * (1.0 - tau)
                new_routing = (predicted_expert_norms >= thresholds).float()
                
                # ** Save the routing mask on the module for later use **
                moe.last_routing_mask = new_routing.clone()

                # Override gate to return the TAU-based mask
                moe.gate = lambda x_in, mask=new_routing: mask

                # Accumulate expert selection statistics.
                total_experts_selected += new_routing.sum().item()
                total_tokens += new_routing.size(0) * new_routing.size(1)

            # --- Second Pass: Forward with TAU-filtered Experts ---
            output = model(label_B, x_BLCv_wo_first_l)  # Shape: (B, L, V)

            # Store last batch for visualization
            last_inp_B3HW = inp_B3HW
            last_logits_BLV = output

            # Track the number of experts selected per token
            experts_per_token_list = []

            for moe in moe_modules:
                if hasattr(moe, "last_routing_mask"):
                    experts_per_token = moe.last_routing_mask.sum(dim=-1)  # Sum over expert dimension
                    experts_per_token_list.append(experts_per_token)

            # Concatenate all expert counts (if multiple MoE layers exist)
            if experts_per_token_list:
                experts_per_token_tensor = torch.cat(experts_per_token_list, dim=-1)  # Shape: (B, L)
                avg_experts_per_token = experts_per_token_tensor.float().mean().item()
                min_experts_per_token = experts_per_token_tensor.float().min().item()
                max_experts_per_token = experts_per_token_tensor.float().max().item()
            else:
                avg_experts_per_token = float('nan')
                min_experts_per_token = float('nan')
                max_experts_per_token = float('nan')

            # Compute losses and accuracies
            L_mean = tc.trainer.val_loss(output.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail = tc.trainer.val_loss(
                output.data[:, -tc.trainer.last_l:].reshape(-1, V),
                gt_BL[:, -tc.trainer.last_l:].reshape(-1)
            ) * B

            acc_mean = (output.data.argmax(dim=-1) == gt_BL).sum() * (100.0 / L)
            acc_tail = (output.data[:, -tc.trainer.last_l:].argmax(dim=-1)
                        == gt_BL[:, -tc.trainer.last_l:]).sum() * (100.0 / tc.trainer.last_l)

            # Print statistics for debugging
            print(f"Avg Experts Per Token: {avg_experts_per_token:.2f}, "
                f"Min: {min_experts_per_token}, Max: {max_experts_per_token}")
            # Accumulate statistics
            total_L_mean += L_mean
            total_L_tail += L_tail
            total_acc_mean += acc_mean
            total_acc_tail += acc_tail
            total_samples += B

            # Restore original gate functions
            for moe in moe_modules:
                moe.gate = original_gates[moe]


    # --- Visualization of Multi-Scale Predictions ---
    if last_inp_B3HW is not None and last_logits_BLV is not None:
        pred_BL = last_logits_BLV.argmax(dim=-1)

        v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        split_sizes = [pn * pn for pn in v_patch_nums]
        scales_tokens = torch.split(pred_BL, split_sizes, dim=1)

        # Get reconstructed images for each scale
        pred_img_B3HW_list = tc.model_vae.idxBl_to_img(
            ms_idx_Bl=scales_tokens,  
            same_shape=False,  
            last_one=False    
        )

        # Ensure idxBl_to_img returned a list of images
        if not isinstance(pred_img_B3HW_list, list):
            pred_img_B3HW_list = [pred_img_B3HW_list]

        # Get experts per token information
        experts_per_token_list = []
        for moe in moe_modules:
            if hasattr(moe, "last_routing_mask"):
                experts_per_token = moe.last_routing_mask.sum(dim=-1)  # Sum over experts
                experts_per_token_list.append(experts_per_token)

        if experts_per_token_list:
            experts_per_token_tensor = torch.cat(experts_per_token_list, dim=-1)  # Shape: (B, L)
        else:
            experts_per_token_tensor = None  # No expert information available

        # Convert first sample to numpy for visualization
        batch_idx = 0  # Pick the first sample
        true_img_3HW = last_inp_B3HW[batch_idx].detach().cpu()
        true_img_np = true_img_3HW.permute(1,2,0).numpy().clip(0,1)

        num_scales = len(pred_img_B3HW_list)
        fig, axes = plt.subplots(2, num_scales + 1, figsize=((num_scales + 1) * 3, 6))  # +1 for true image

        # Plot all predicted scales in the top row
        for i, pred_img_3HW in enumerate(pred_img_B3HW_list):
            pred_img_np = pred_img_3HW[batch_idx].detach().cpu().permute(1,2,0).numpy().clip(0,1)
            
            axes[0, i].imshow(pred_img_np)
            axes[0, i].set_title(f"{v_patch_nums[i]*16}x{v_patch_nums[i]*16}")  # Scale size
            axes[0, i].axis("off")

        # Add the **ground truth** final image at the end
        axes[0, num_scales].imshow(true_img_np)
        axes[0, num_scales].set_title("Ground Truth (Final)")
        axes[0, num_scales].axis("off")

        # Plot expert usage per token below each predicted image
        if experts_per_token_tensor is not None:
            experts_per_token_np = experts_per_token_tensor[batch_idx].detach().cpu().numpy()

            for i in range(num_scales):
                token_counts = experts_per_token_np[:split_sizes[i]].reshape(v_patch_nums[i], v_patch_nums[i])
                axes[1, i].imshow(token_counts, cmap="viridis", aspect="auto")
                axes[1, i].set_title("Experts Used")
                axes[1, i].axis("off")
                for (m, n), val in np.ndenumerate(token_counts):
                    axes[1, i].text(n, m, int(val), ha='center', va='center', color='white')

        # Add a blank spot below the **ground truth** image (since it has no expert data)
        axes[1, num_scales].axis("off")

        # Save the visualization
        plt.tight_layout()
        plt.savefig(f"plots_with_tau/multi_scale_experts_{tau}.png")
        plt.close(fig)


    return total_L_mean, total_L_tail, total_acc_mean, total_acc_tail, total_experts_selected / total_tokens

def capturing_gate_for_moe(moe, x_in):
    """Return a gate function that sets moe.last_router_input and returns an all-ones routing tensor."""
    def _capturing_gate(x):
        moe.last_router_input = x
        return torch.ones(
            x.size(0), x.size(1), moe.num_experts,
            dtype=x.dtype, device=x.device
        )
    return _capturing_gate


# 1) Force all experts and hook only the final sub-layer of each expert.
# Correct
def attach_final_expert_hooks(moe_modules, x_in):
    def expert_output_hook(module, input, output):
        # Save the output of this expert's final sub-layer
        module.last_expert_output = output

    for moe in moe_modules:
        # Overwrite the gate with a function that returns all ones
        moe.gate = capturing_gate_for_moe(moe, x_in)
        
        # Attach the forward hook only to the final sub-layer
        final_sub_layer = moe.experts.layers[-1]
        final_sub_layer.register_forward_hook(expert_output_hook)


def compute_tau_routing(moe, tau, x_shape, x):
    """Compute the L2 norm of each expert's final output and build a binary TAU-based mask."""
    
    expert_outputs = moe.experts.layers[-1].last_expert_output


    # Now compute the L2 norms:
    # shape: (num_experts, B*T) or (e, n)
    norms = torch.linalg.vector_norm(expert_outputs, ord=2, dim=-1)

    # Suppose B = input.size(0), T = input.size(1)
    e = norms.size(0)
    B = x_shape[0]
    T = x_shape[1]

    max_norms = norms.view(e, B, T).permute(1, 2, 0)  # => shape (B, T, e)
    
    max_norms, _ = norms.max(dim=-1, keepdim=True)
    norm_thresholds = max_norms * (1.0 - tau)
    
    new_routing = torch.zeros_like(norms)
    new_routing[norms >= norm_thresholds] = 1.0
    new_routing = new_routing.transpose(0, 1) 
    
    # Save the mask and override gate:
    moe.last_routing_mask = new_routing.clone()
    moe.gate = lambda x_in, mask=new_routing: mask
    
    usage_per_token = new_routing.sum(dim=-1)  # shape (B, T)
    moe.last_expert_usage = usage_per_token.clone()
    


def run_inference_with_tau(tc, x, moe_modules, cond_BD_or_gss, tau, debug_data, si):
    """
    Helper to run the second pass with the TAU-based gating in the MoE model (tc.model)
    """
    list_of_outputs = []
    for index, b in enumerate(tc.model.blocks):
        # 1) Build TAU-based routing for the current MoE block
        compute_tau_routing(moe_modules[index], tau, x.shape, x)
        
        # 2) Identify the corresponding block in the base model
        moeblock  = tc.model.blocks[index]
        out = moeblock(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        x = out[0] if isinstance(out, tuple) else out
        list_of_outputs.append(x.clone())

    return x, list_of_outputs

def max_difference(expected, got):
    """
    Compute the maximum absolute difference between two tensors or arrays.
    Both expected and got should be either numpy arrays or torch.Tensors.
    If they are torch.Tensors, they are moved to CPU and converted to numpy arrays.
    """
    expected_np = expected.cpu().numpy() if isinstance(expected, torch.Tensor) else expected
    got_np = got.cpu().numpy() if isinstance(got, torch.Tensor) else got
    return np.abs(expected_np - got_np).max()


@torch.no_grad()
def autoregressive_infer_cfg_with_expert_plot(
    tc,
    B: int,
    label_B: Optional[Union[int, torch.LongTensor]], 
    g_seed: Optional[int] = None, 
    cfg: float = 1.5, 
    top_k: int = 0, 
    top_p: float = 0.0,
    rng = 0,
    more_smooth: bool = False,
    tau: float = 1.0,
    debug_data = None, 
    compare_dicts: bool = False,  # Set to True to collect debug info and perform comparisons.
    type_of_model: str = "MoE_no_FT",
    final_path_save: str = "data"
) -> torch.Tensor:
    """
    Autoregressive inference with CFG and two-pass TAU-based expert selection.
    
    If compare_dicts is True, the function will:
      - Construct a predicted_dictionary collecting intermediate tensors.
      - Perform debug comparisons against debug_data.
      - Save the predicted_dictionary to a pickle file.
    
    Otherwise, the function will only compute and return the final image.
    """
    if g_seed is None: 
        rng = None
    else: 
        self.rng.manual_seed(g_seed)
        rng = self.rng

    if label_B is None:
        label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

    # Prepare conditioning tokens and positional embeddings.
    cond_BD = sos = tc.model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=tc.model.num_classes)), dim=0))
    lvl_pos = tc.model.lvl_embed(tc.model.lvl_1L) + tc.model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, tc.model.first_l, -1) \
                     + tc.model.pos_start.expand(2 * B, tc.model.first_l, -1) \
                     + lvl_pos[:, :tc.model.first_l]

    # Initialize latent representation.
    f_hat = sos.new_zeros(B, tc.model.Cvae, tc.model.patch_nums[-1], tc.model.patch_nums[-1])

    for b in tc.model.blocks:
        b.attn.kv_caching(True)
        b.ffn.forward_mode = 'oracle'
        b.ffn.tau = tau  

    # Get all MoE modules within the blocks.
    moe_modules = [m for b in tc.model.blocks for m in b.modules() if hasattr(m, 'gate') and hasattr(m, 'router')]
    original_gates = {m: m.gate for m in moe_modules}

    cur_L = 0
    num_scales = len(tc.model.patch_nums)

    # Only construct the debug dictionary if requested.
    if compare_dicts:
        predicted_dictionary = {
            "f_hat": [],
            "idx_Bl": [],
            "h_BChw": [],
            "logits_BlV": [],
            "ratio": [],
            'next_token_map': [],
            'x': [],
            'cond_BD_or_gss': [],
            'block_output': [],
            'experts_per_token': [],
            'img': [],    
        }

    # Iterate over scales.
    total_flops = 0
    mean_flops = 0
    for si, pn in enumerate(tc.model.patch_nums):
        ratio = si / tc.model.num_stages_minus_1
        if compare_dicts:
            predicted_dictionary['ratio'].append(ratio)
        cur_L += pn * pn

        # Compute conditioning for this scale.
        cond_BD_or_gss = tc.model.shared_ada_lin(cond_BD)
        if compare_dicts:
            predicted_dictionary['cond_BD_or_gss'].append(cond_BD_or_gss.detach().cpu().clone())

        x = next_token_map  # Autoregressive token input.
        list_of_outputs = []
        for index, b in enumerate(tc.model.blocks):  
            out = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            x = out[0] if isinstance(out, tuple) else out
            if compare_dicts:
                list_of_outputs.append(x.clone())
       
        if compare_dicts:
            predicted_dictionary['x'].append(x.detach().cpu().clone())
            predicted_dictionary['block_output'].append(list_of_outputs)
            
            usage_list_for_this_scale = []
            for idx in range(len(moe_modules)-1):
                moex = moe_modules[idx]
                usage_list_for_this_scale.append(moex.routing_mask.clone())  
            predicted_dictionary['experts_per_token'].append(usage_list_for_this_scale)
        
        total_flops += sum(moex.routing_mask.clone().sum() for moex in moe_modules)
        mean_flops += sum(moex.routing_mask.clone().mean() for moex in moe_modules)

        logits_BlV = tc.model.get_logits(x, cond_BD)
        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        if compare_dicts:
            predicted_dictionary['logits_BlV'].append(logits_BlV.detach().cpu().clone())

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        if compare_dicts:
            predicted_dictionary['idx_Bl'].append(idx_Bl.detach().cpu().clone())
       
        # --- Update latent representation ---        
        h_BChw = tc.model.vae_quant_proxy[0].embedding(idx_Bl)
        if compare_dicts:
            predicted_dictionary['h_BChw'].append(h_BChw.detach().cpu().clone())

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, tc.model.Cvae, pn, pn)

        f_hat, next_token_map = tc.model.vae_quant_proxy[0].get_next_autoregressive_input(si, num_scales, f_hat, h_BChw)
        if compare_dicts:
            predicted_dictionary['f_hat'].append(f_hat.clone())

        final_img = tc.model_vae.fhat_to_img(f_hat.clone())
        img = final_img[0].add_(1).mul_(0.5).permute(1, 2, 0).mul(255).clamp(0,255).cpu().numpy().astype(np.uint8)
        if compare_dicts:
            predicted_dictionary['img'].append(img)

        if si != tc.model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, tc.model.Cvae, -1).transpose(1, 2)
            next_token_map = tc.model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + tc.model.patch_nums[si + 1] ** 2]     
            next_token_map = next_token_map.repeat(2, 1, 1)
            if compare_dicts:
                predicted_dictionary['next_token_map'].append(next_token_map.detach().cpu().clone())

    for b in tc.model.blocks:
        b.attn.kv_caching(False)

    # If debug comparisons were requested, run them.
    if compare_dicts and debug_data is not None:
        for si, pn in enumerate(tc.model.patch_nums):
            print()
            print("-"*50)
            print(f"[DEBUG] Scale {si} started.")
            print("-"*50)
            print()

            if np.allclose(debug_data['ratio'][si], predicted_dictionary['ratio'][si]):
                print(f"[DEBUG] ratio for scale {si} are correct.")
            else:
                print(f"[DEBUG] ratio for scale {si} are incorrect.")
                print(f"[DEBUG] Expected: {debug_data['ratio'][si]}")
                print(f"[DEBUG] Got: {predicted_dictionary['ratio'][si]}")

            if np.allclose(debug_data['cond_BD_or_gss'][si].cpu().numpy(), predicted_dictionary['cond_BD_or_gss'][si].cpu().numpy()):
                print(f"[DEBUG] cond_BD_or_gss for scale {si} are correct.")
            else:
                print(f"[DEBUG] cond_BD_or_gss for scale {si} are incorrect.")
                print(f"[DEBUG] Expected: {debug_data['cond_BD_or_gss'][si]}, sum of expected: {debug_data['cond_BD_or_gss'][si].sum()}")
                print(f"[DEBUG] Got: {predicted_dictionary['cond_BD_or_gss'][si].cpu().numpy()}, sum of got: {predicted_dictionary['cond_BD_or_gss'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['cond_BD_or_gss'][si], predicted_dictionary['cond_BD_or_gss'][si].cpu().numpy())}")

            if np.allclose(debug_data['x'][si].cpu().numpy(), predicted_dictionary['x'][si].cpu().numpy()):
                print(f"[DEBUG] x for scale {si} are correct.")
            else:
                print(f"[DEBUG] x for scale {si} are incorrect.")
                print(f"[DEBUG] sum of expected: {debug_data['x'][si].sum()}")
                print(f"[DEBUG] sum of got: { predicted_dictionary['x'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['x'][si],  predicted_dictionary['x'][si].cpu().numpy())}")
            
            if np.allclose(debug_data['logits_BlV'][si].cpu().numpy(), predicted_dictionary['logits_BlV'][si].cpu().numpy()):
                print(f"[DEBUG] logits_BlV for scale {si} are correct.")
            else:
                print(f"[DEBUG] logits_BlV for scale {si} are incorrect.")
                print(f"[DEBUG] sum of expected: {debug_data['logits_BlV'][si].sum()}")
                print(f"[DEBUG] sum of got: {predicted_dictionary['logits_BlV'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['logits_BlV'][si], predicted_dictionary['logits_BlV'][si].cpu().numpy())}")

            if np.allclose(debug_data['idx_Bl'][si].cpu().numpy(), predicted_dictionary['idx_Bl'][si].cpu().numpy()):
                print(f"[DEBUG] idx_Bl for scale {si} are correct.")
            else:
                print(f"[DEBUG] idx_Bl for scale {si} are incorrect.")
                print(f"[DEBUG] sum of expected: {debug_data['idx_Bl'][si].sum()}")
                print(f"[DEBUG] sum of got: {predicted_dictionary['idx_Bl'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['idx_Bl'][si], predicted_dictionary['idx_Bl'][si].cpu().numpy())}")

            if np.allclose(debug_data['h_BChw'][si].cpu().numpy(), predicted_dictionary['h_BChw'][si].cpu().numpy()):
                print(f"[DEBUG] h_BChw for scale {si} are correct.")
            else:
                print(f"[DEBUG] h_BChw for scale {si} are incorrect.")
                print(f"[DEBUG] sum of expected: {debug_data['h_BChw'][si].sum()}")
                print(f"[DEBUG] sum of got: {predicted_dictionary['h_BChw'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['h_BChw'][si], predicted_dictionary['h_BChw'][si].cpu().numpy())}")

            for index in range(16):
                if np.allclose(debug_data['block_output'][si][index].cpu().numpy(), predicted_dictionary['block_output'][si][index].cpu().numpy()):
                    print(f"[DEBUG] x for scale {si} for block {index} are correct.")
                else:
                    print(f"[DEBUG] x for scale {si} for block {index} are incorrect.")
                    print(f"[DEBUG]  sum of expected: {debug_data['block_output'][si][index].sum()}")
                    print(f"[DEBUG]  sum of got: { predicted_dictionary['block_output'][si][index].sum()}")

            if np.allclose(debug_data['f_hat'][si].cpu().numpy(), predicted_dictionary['f_hat'][si].cpu().numpy()):
                print(f"[DEBUG] f_hat for scale {si} are correct.")
            else:
                print(f"[DEBUG] f_hat for scale {si} are incorrect.")
                print(f"[DEBUG] sum of expected: {debug_data['f_hat'][si].sum()}")
                print(f"[DEBUG] sum of got: {predicted_dictionary['f_hat'][si].cpu().numpy().sum()}")
                print(f"[DEBUG] max difference: {max_difference(debug_data['f_hat'][si], predicted_dictionary['f_hat'][si].cpu().numpy())}")

            if si < len(tc.model.patch_nums) - 1:
                if np.allclose(debug_data['next_token_map'][si].cpu().numpy(), predicted_dictionary['next_token_map'][si].cpu().numpy()):
                    print(f"[DEBUG] next_token_map for scale {si} are correct.")
                else:
                    print(f"[DEBUG] next_token_map for scale {si} are incorrect.")
                    print(f"[DEBUG] sum of expected: {debug_data['next_token_map'][si].sum()}")
                    print(f"[DEBUG] sum of got: {predicted_dictionary['next_token_map'][si].cpu().numpy().sum()}")
                    print(f"[DEBUG] max difference: {max_difference(debug_data['next_token_map'][si], predicted_dictionary['next_token_map'][si].cpu().numpy())}")
            
            print()
            print("-"*50)
            print(f"[DEBUG] Scale {si} completed.")
            print("-"*50)
            print()

    # Save the debug dictionary only if requested.
    if compare_dicts:
        os.makedirs(final_path_save, exist_ok=True)
        with open(f"{final_path_save}/{type_of_model}_with_tau_{tau}.pkl", "wb") as f:
            pickle.dump(predicted_dictionary, f)
            print(f"[DEBUG] Saved the data to {final_path_save}/{type_of_model}_with_tau_{tau}.pkl")
        with open(f"{final_path_save}/base_model.pkl", "wb") as f:
            pickle.dump(debug_data, f)

    # Restore original MoE gate functions.
    for m in moe_modules:
        m.gate = original_gates[m]

    

    return tc.model_vae.fhat_to_img(f_hat).add_(1).mul_(0.5), mean_flops, total_flops




def benchmark_moe(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader, tc=None) -> Tuple[float, Dict, Dict]:
    from architectures.moe.moe_layers import MoELayer, ExecuteAllExperts, ModuleBatchedExperts, CustomKernelExperts
    model_costs, model_params = benchmark(model, data_loader, tc)

    # Find MoE layers
    moe_module_names = find_module_names(model, lambda _, m: isinstance(m, MoELayer))

    # Find expert modules inside each MoE layer
    experts_module_names = {}
    for moe_module_name in moe_module_names:
        moe_module = get_module_by_name(model, moe_module_name)
        experts_names = find_module_names(moe_module, lambda _, m: isinstance(m, (ExecuteAllExperts, ModuleBatchedExperts, CustomKernelExperts)))
        assert len(experts_names) == 1, f'{len(experts_names)=}'
        experts_module_names[moe_module_name] = f'{moe_module_name}.{experts_names[0]}'

    # Add hooks to capture activations
    expert_module_name_list = list(experts_module_names.values())
    experts_inputs, _, experts_handles = add_save_activations_hook(model, expert_module_name_list)

    # Run a forward pass with a single batch
    X, y = next(iter(data_loader))
    if isinstance(X, dict):
        sample = {k: v[:1] for k, v in X.items()}
    else:
        sample = X[:1]

    with torch.no_grad():
        B, V = y.shape[0], tc.model_vae.vocab_size
        X = X.to(dist.get_device(), non_blocking=True)
        label_B = y.to(dist.get_device(), non_blocking=True)
        gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X)
        x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)

    # Run model inference
    model(label_B, x_BLCv_wo_first_l).detach()

    # Initialize cost tracking
    cost_without_experts = model_costs.total()
    expert_costs = {}
    expert_selection_stats = {}
    experts_per_token_stats = {}
    total_experts_used = 0
    total_blocks = len(moe_module_names)

    # Process each MoE layer
    for moe_name in moe_module_names:
        experts_name = experts_module_names[moe_name]
        experts_module = get_module_by_name(model, experts_name)

        # Extract routing activations (expert selection probabilities)
        routing_tensor = experts_inputs[experts_name][1]  # Shape: [num_tokens, num_experts]
        num_experts = experts_module.num_experts
  
        # Compute expert selection distribution
        expert_counts = routing_tensor.argmax(dim=-1).bincount(minlength=num_experts).float()
        expert_selection_percentages = (expert_counts / expert_counts.sum()) * 100

        # Compute how many experts are selected per token
        experts_per_token = (routing_tensor > 0).sum(dim=-1)  # Counts nonzero entries per token
        avg_experts_per_token = experts_per_token.float().mean().item()
        total_experts_used += avg_experts_per_token

        # Store the stats for logging
        expert_selection_stats[moe_name] = expert_selection_percentages.tolist()
        experts_per_token_stats[moe_name] = avg_experts_per_token

        # Compute gating cost
        gating_cost = model_costs.by_module()[moe_name] - model_costs.by_module()[experts_name]

        # Compute per-token expert cost
        experts_input = experts_inputs[experts_name]
        device, dtype = experts_input[1].device, experts_input[1].dtype
        dummy_routing_tensor = torch.zeros((1, num_experts), device=device, dtype=dtype)
        dummy_routing_tensor[0, 0] = 1.0  # Only one expert is selected for benchmarking

        experts_input = (experts_input[0][:1], dummy_routing_tensor)
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            token_expert_cost = flop_count(experts_module, experts_input).total()

        if isinstance(experts_module, (ExecuteAllExperts, CustomKernelExperts)):
            token_expert_cost /= num_experts  # Normalize for MoE models where all experts run

        cost_without_experts -= model_costs.by_module()[moe_name] - gating_cost
        expert_costs[moe_name] = token_expert_cost

    remove_hooks(experts_handles)

    # Compute average experts per token across all blocks
    avg_experts_across_blocks = total_experts_used / total_blocks if total_blocks > 0 else 0

    logging.info(f'Average experts per token across all blocks: {avg_experts_across_blocks:.2f}')
    logging.info(f'Model cost without experts: {cost_without_experts}')

    return cost_without_experts, expert_costs, model_params

def benchmark_avit(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader):
    X, _ = next(iter(data_loader))
    device = X.device
    del X
    model_costs, model_params = benchmark(model, data_loader)
    total_seq_len = model.num_total_tokens
    constant_cost = model_costs.total()
    # warning - assumes that all the MHA and MLP modules are the same!
    mha_module_names = find_module_names(model, lambda _, m: isinstance(m, MultiheadAttention))
    for mha_module_name in mha_module_names:
        constant_cost -= model_costs.by_module()[mha_module_name]
    # save MHA cost for each possible sequence length
    mha_sequence_costs = [0]
    mha_module = get_module_by_name(model, mha_module_name)
    for i in range(total_seq_len):
        seq_len = i + 1
        dummy_input = torch.randn(1, seq_len, model.hidden_dim, device=device)
        dummy_input = (dummy_input, dummy_input, dummy_input)
        mha_cost_for_seq = flop_count(mha_module, dummy_input).total()
        mha_sequence_costs.append(mha_cost_for_seq)
    # save MLP cost for a single token
    dummy_input = torch.randn(1, 1, model.hidden_dim, device=device)
    mlp_module_names = find_module_names(model, lambda _, m: isinstance(m, MLP) or isinstance(m, CustomMLP))
    for mlp_module_name in mlp_module_names:
        constant_cost -= model_costs.by_module()[mlp_module_name]
    mlp_module = get_module_by_name(model, mlp_module_name)
    # save LN cost for a single token
    ln_module_names = find_module_names(model, lambda _, m: isinstance(m, LayerNorm))
    for ln_module_name in ln_module_names:
        constant_cost -= model_costs.by_module()[ln_module_name]
    ln_module = get_module_by_name(model, ln_module_name)
    mlp_ln_token_cost = flop_count(mlp_module, dummy_input).total() + 2 * flop_count(ln_module, dummy_input).total()
    return constant_cost, mha_sequence_costs, mlp_ln_token_cost, model_params
