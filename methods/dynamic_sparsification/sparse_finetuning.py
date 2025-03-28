import logging
from datetime import datetime
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from torch import nn
from torchvision.ops import MLP as TorchvisionMLP
from transformers.activations import GELUActivation, PytorchGELUTanh
from transformers.models.bert.modeling_bert import BertIntermediate
from transformers.models.gemma.modeling_gemma import GemmaMLP

from architectures.gpt import MLP as GPTMLP
from architectures.moe.dsti import SimpleMLP, find_relu_activations
from architectures.vit import MLP
from common import INIT_NAME_MAP, get_default_args
from train import (
    TrainingContext,
    final_eval,
    in_training_eval,
    setup_accelerator,
    setup_data,
    setup_files_and_logging,
    setup_optimization,
    setup_state,
    make_vae,
)
from utils import (
    Mixup,
    add_save_activations_hook,
    find_module_names,
    get_lrs,
    get_module_by_name,
    get_module_name,
    get_parent_module_name,
    load_model,
    save_state,
    set_module_by_name,
    save_final,
)
from trainer import VARTrainer
from utils_var import arg_util
from utils_var.misc import auto_resume
from architectures import VAR, VQVAE, build_vae_var
from architectures.basic_var import FFN
import dist
from pathlib import Path
from architectures.pretrained import get_var_d16
from collections import OrderedDict


class EnforceSparsityTrainingContext(TrainingContext):
    sparsity_enforcement_mode: str = None
    modules_inputs: Dict = None
    modules_outputs: Dict = None
    model_hook_handles: List = None
    var_wo_ddp = None


def eligible_activation_filter(model: nn.Module, m: nn.Module):
    # TODO handle cases when functional variant is used instead
    m_name = get_module_name(model, m)
    parent_module = get_module_by_name(model, get_parent_module_name(m_name))
    # print("----"*100)
    # print("m_name", m_name) 
    # print("parent", parent_module)
    # print(isinstance(m, (nn.ReLU, nn.GELU, GELUActivation, PytorchGELUTanh)), isinstance(parent_module, (MLP, TorchvisionMLP, SimpleMLP, GPTMLP, BertIntermediate, GemmaMLP, FFN)))
    # print("----"*100)
    if isinstance(m, (nn.ReLU, nn.GELU, GELUActivation, PytorchGELUTanh)) and isinstance(parent_module, (
            MLP, TorchvisionMLP, SimpleMLP, GPTMLP, BertIntermediate, GemmaMLP, FFN)):
        return True


def find_activations(model, apply_to):
    if apply_to == 'moe_eligible_only':
        acts_to_sparsify = find_module_names(model, eligible_activation_filter)
    elif apply_to == 'everywhere':
        acts_to_sparsify = find_module_names(model, lambda _, m: isinstance(m, (
        nn.ReLU, nn.GELU, GELUActivation, PytorchGELUTanh)))
    else:
        raise ValueError(f'Invalid apply_to argument value: {apply_to}')
    return acts_to_sparsify


def replace_with_relu(model, acts_to_replace):
    for act_m_name in acts_to_replace:
        logging.info(f'Replacing {act_m_name} with ReLU')
        set_module_by_name(model, act_m_name, nn.ReLU())
    return model


def setup_model(args, tc):
    assert args.model_class == 'enforce_sparsity'
    model, tc.model_vae = get_var_d16()
    activations_to_sparsify = find_activations(model, **args.model_args)

    if 'relu' in args.dsti_enforce_mode:
        model = replace_with_relu(model, activations_to_sparsify)
        # TODO add "apply_to" argument to find_relu_activations
        activations_to_sparsify.extend(find_relu_activations(model))
    logging.info(f'Modules selected for activation sparsification: {activations_to_sparsify}')


    if args.path_file_ft:
        init_path = Path(args.path_file_ft)
        final_state = torch.load(init_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model = model.to(args.device)
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)

    tc.modules_inputs, tc.modules_outputs, tc.model_hook_handles = \
        add_save_activations_hook(model, activations_to_sparsify)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = model
    # make sure all parameters are being optimized
    # tc.model.requires_grad_(True)


def square_hoyer(tensor, dim=-1, eps=1e-15):
    """As in http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf Section 2.1"""
    return torch.linalg.vector_norm(tensor, ord=1.0, dim=dim) ** 2 / \
        (torch.linalg.vector_norm(tensor, ord=2.0, dim=dim) ** 2 + eps)


def training_loop(args, tc):
    print('Start training loop')
    model_saved = datetime.now()
    train_iter = iter(tc.train_loader)

    print("Model parameters being trained:")
    for name, param in tc.model.named_parameters():
        if 'blocks' in name or 'head' in name:
            param.requires_grad = True

    print(" ".join(
        f"Parameter: {name}, Shape: {param.shape}"
        for name, param in tc.model.named_parameters() if param.requires_grad
    ))

    train_loss = nn.CrossEntropyLoss(label_smoothing=args.ls, reduction='none')
    val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
    L = sum(pn * pn for pn in args.patch_nums)
    loss_weight = torch.ones(1, L, device=args.device) / L


    mixup_fn = None

    current_batch = 0
    accum_acc = 0.0
    accum_grad_norm = 0.0
    accum_task_loss = 0.0
    accum_dsti_weight = 0.0
    accum_sparsity_loss_ffn = 0.0
    counter = 0
    while current_batch < tc.last_batch:
        # save model conditionally

        now = datetime.now()
        if (now - model_saved).total_seconds() > 60*60:
            (L_mean, L_tail, acc_mean, acc_tail, tot, time, model_costs, param_count_dict) = tc.trainer.eval_ep(tc.val_loader)
            print(
                f"Model Performance: "
                f"L_mean={L_mean:.4f}, "
                f"L_tail={L_tail:.4f}, "
                f"acc_mean={acc_mean:.2f}%, "
                f"acc_tail={acc_tail:.2f}%, "
                f"tot={tot}, "
                f"time={time:.2f}s"
            )
            if isinstance(args.runs_dir, str):
                args.runs_dir = Path("runs")
            args.runs_dir.mkdir(parents=True, exist_ok=True)
            run_name = str(args.final_path_save) + "_" + str(args.dsti_enforce_weight)
            run_dir = args.runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            tc.final_path = run_dir / 'final.pth'

            model_saved = datetime.now()
            final_results = {}
            final_results['args'] = args

            for module in tc.model.modules():
                module._forward_hooks.clear()
                module._forward_pre_hooks.clear()
                module._backward_hooks.clear()
                module._backward_pre_hooks.clear()

            
            final_results['model_state'] = tc.model.state_dict()

            final_results['L_mean'] = L_mean
            final_results['L_tail'] = L_tail
            final_results['acc_mean'] = acc_mean
            final_results['acc_tail'] = acc_tail
            final_results['tot'] = tot
            final_results['time']= time

            save_final(args, tc.final_path, final_results)

        # model evaluation
        #in_training_eval(args, tc)
        tc.model.train()
        tc.optimizer.zero_grad(set_to_none=True)
        # Account for gradient accumulation
        # cumulative_task_loss = 0
        # cumulative_sparsity_loss_ffn = 0
        # cumulative_sparsity_loss_attn = 0
        # cumulativte_total_loss = 0


        for _ in range(args.gradient_accumulation_steps):
            # batch preparation
            X, y = next(train_iter)

            # forward
            with torch.no_grad():
                B, V = y.shape[0], tc.model_vae.vocab_size
                X = X.to(dist.get_device(), non_blocking=True)
                label_B = y.to(dist.get_device(), non_blocking=True)
                gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X) # This does not return None
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            
            logits_BLV = tc.model(label_B, x_BLCv_wo_first_l)
            # get activations / pre-activations and compute sparsity loss
            sparsity_loss_ffn = 0.0
            sparse_activations_sum_ffn = 0
            total_activations_sum_ffn = 0
            total_num_outputs_ffn = 0
            sparsity_loss_attn = 0.0
            sparse_activations_sum_attn = 0
            total_activations_sum_attn = 0
            total_num_outputs_attn = 0
            # warning! inputs (e.g. pre-activations for gelu) are tuples, outputs (e.g. relu activations) are tensors
            if args.dsti_enforce_mode == 'gelu_preactivations_hoyer_clamped':
                for module_name, preacts in tc.modules_inputs.items():
                    # get first input tensor from the input tuple
                    assert isinstance(preacts, tuple)
                    preacts = preacts[0]
                    clamped_preactivations = preacts.clamp(
                        min=args.dsti_clamp_displacement) - args.dsti_clamp_displacement
                    sparsity_loss = square_hoyer(clamped_preactivations, dim=-1).mean()
                    sparse_activations_sum = (clamped_preactivations <= args.dsti_clamp_displacement).sum().item()
                    total_activations_sum = preacts.numel()
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name:
                        continue
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            elif args.dsti_enforce_mode == 'relu_hoyer':
                for module_name, acts in tc.modules_outputs.items():
                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name: # Jort: For now we skip this.
                        continue
                    assert isinstance(acts, torch.Tensor)
                    sparsity_loss = square_hoyer(acts, dim=-1).mean()
                    sparse_activations_sum = (acts <= 0).sum().item()
                    total_activations_sum = acts.numel()

                    if 'q_proj' in module_name or 'k_proj' in module_name or 'v_proj' in module_name or 'o_proj' in module_name: # Jort: For now we skip this.
                        sparsity_loss_attn += sparsity_loss
                        sparse_activations_sum_attn += sparse_activations_sum
                        total_activations_sum_attn += total_activations_sum
                        total_num_outputs_attn += 1
                    else:
                        assert 'ffn' in module_name
                        sparsity_loss_ffn += sparsity_loss
                        sparse_activations_sum_ffn += sparse_activations_sum
                        total_activations_sum_ffn += total_activations_sum
                        total_num_outputs_ffn += 1
            else:
                raise ValueError(f'{args.enforce_mode=}')

            if total_num_outputs_attn > 0:
                sparsity_loss_attn /= total_num_outputs_attn
            sparsity_loss_ffn /= total_num_outputs_ffn
            # task loss
            task_loss = train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            task_loss = task_loss.mul(loss_weight).sum(dim=-1).mean()

            
            if args.dsti_enforce_schedule == 'linear':
                dsti_enforce_weight = current_batch / tc.last_batch * args.dsti_enforce_weight
            elif args.dsti_enforce_schedule == 'linear_warmup':
                # For the first 1/10 of training, weight = 0
                # Then ramp linearly up to 1 * args.dsti_enforce_weight by last_batch/2
                start_ramp_batch = int(tc.last_batch / 10)   # 10% of total
                end_ramp_batch   = int(tc.last_batch / 2)    # 50% of total

                if current_batch < start_ramp_batch:
                    # First 10% => enforce weight is zero
                    dsti_enforce_weight = 0.0
                elif current_batch < end_ramp_batch:
                    # From 10% to 50% => linearly ramp from 0 to 1
                    progress = (current_batch - start_ramp_batch) / float(end_ramp_batch - start_ramp_batch)
                    progress = max(progress, 0.0)  # just to be safe
                    progress = min(progress, 1.0)
                    dsti_enforce_weight = progress * args.dsti_enforce_weight
                else:
                    # After 50% => full weight
                    dsti_enforce_weight = args.dsti_enforce_weight
            else:
                dsti_enforce_weight = args.dsti_enforce_weight
            # loss computation
            loss = (task_loss +
                    dsti_enforce_weight * sparsity_loss_ffn +
                    dsti_enforce_weight * sparsity_loss_attn)


            grad_norm, scale_log2 = tc.trainer.var_opt.backward_clip_step(loss=loss, stepping=True)

            pred_BL = logits_BLV.data.argmax(dim=-1)
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            grad_norm = grad_norm.item()
            
            accum_acc += acc_mean
            accum_grad_norm += grad_norm
            accum_task_loss += task_loss
            accum_dsti_weight += dsti_enforce_weight
            accum_sparsity_loss_ffn += sparsity_loss_ffn
            counter += 1

            # Every 10 iterations, print the average and reset accumulators
            if counter % 10 == 0:
                avg_acc = accum_acc / counter
                avg_grad_norm = accum_grad_norm / counter
                avg_task_loss = accum_task_loss / counter
                avg_dsti_weight = accum_dsti_weight / counter
                avg_sparsity_loss_ffn = accum_sparsity_loss_ffn / counter

                print(f'Completion rate: {current_batch/tc.last_batch:.2f}% | '
                    f'Acc: {avg_acc:.2f} | '
                    f'Grad norm: {avg_grad_norm:.2f} | '
                    f'task_loss: {avg_task_loss:.4f} | '
                    f'dsti_weight: {avg_dsti_weight:.4f} | '
                    f'sparsity_loss_ffn: {avg_sparsity_loss_ffn:.4f}')

                # Reset accumulators and counter
                accum_completion_rate = 0.0
                accum_acc = 0.0
                accum_grad_norm = 0.0
                accum_task_loss = 0.0
                accum_dsti_weight = 0.0
                accum_sparsity_loss_ffn = 0.0
                counter = 0
            
            # Compare parameter changes and assert that they have been updated.
            # for name, param in tc.model.named_parameters():
            #     if param.requires_grad:
            #         assert not torch.equal(old_params[name], param), f"Parameter {name} was expected to update but did NOT change."

            
        #     cumulativte_total_loss += loss.item()
        #     cumulative_task_loss += task_loss.item()
        #     cumulative_sparsity_loss_ffn += float(sparsity_loss_ffn)
        #     cumulative_sparsity_loss_attn += float(sparsity_loss_attn)



        # sparsity_ffn = sparse_activations_sum_ffn / total_activations_sum_ffn
        # if total_num_outputs_attn > 0.:
        #     sparsity_attn = sparse_activations_sum_attn / total_activations_sum_attn
        #     sparsity_total = (sparse_activations_sum_ffn + sparse_activations_sum_attn) / (
        #             total_activations_sum_ffn + total_activations_sum_attn)


        current_batch += 1


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info('Configured logging')
    args: arg_util.Args = arg_util.init_dist_and_get_args(args)
    tc = EnforceSparsityTrainingContext()
    tc.sparsity_enforcement_mode = args.dsti_enforce_mode
    tc.sparsity_enforcement_displacement = args.dsti_clamp_displacement
    #setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_data(args, tc)
    setup_model(args, tc)
    setup_optimization(args, tc)
    #setup_state(tc)
    

    # Build the trainer
    tc.trainer = VARTrainer(
        device=args.device, # correct
        patch_nums=args.patch_nums, # correct
        resos=args.resos, # correct
        vae_local=tc.model_vae,
        var_wo_ddp=tc.model, # correct
        var=tc.model, # correct
        var_opt=tc.optimizer, # correct
        label_smooth=args.ls # correct
    )
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # print(trainer_state)
    # if trainer_state is not None and len(trainer_state):
    #     trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again

    training_loop(args, tc)
    final_eval(args, tc)



def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
