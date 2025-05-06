import logging
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union

import torch
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
from sklearn.metrics import roc_auc_score
from torch.nn import MultiheadAttention, LayerNorm
from architectures import VAR
from utils import flop_count, get_module_by_name, remove_hooks, find_module_names, add_save_activations_hook
from architectures.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

import numpy as np
import dist
import pickle
import os


def test_classification(accelerator: Accelerator,
                        model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        criterion_class: torch.nn.Module,
                        tc, 
                        batches: int = 0) -> Tuple[float, float]:

    # model = accelerator.unwrap_model(model)
    L_mean, L_tail, acc_mean, acc_tail, total, elapsed_time = tc.trainer.eval_ep(tc.val_loader)
    mean_loss = L_mean
    accuracy = acc_mean
    return mean_loss, accuracy



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
    save_sample: bool = False,  # Set to True to collect debug info and perform comparisons.
    forward_mode: str = 'oracle',
    taus = False, 
    expert_index_switch = 0,
    save_routing_pattern = False
) -> torch.Tensor:
    """
    Autoregressive inference with CFG and two-pass TAU-based expert selection.
    
    If compare_dicts is True, the function will:
      - Construct a predicted_dictionary collecting intermediate tensors.
      - Save the predicted_dictionary to a pickle file.
    
    Otherwise, the function will only compute and return the final image.
    """

    # Prepare conditioning tokens and positional embeddings.
    cond_BD = sos = tc.model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=tc.model.num_classes)), dim=0))
    lvl_pos = tc.model.lvl_embed(tc.model.lvl_1L) + tc.model.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, tc.model.first_l, -1) \
                     + tc.model.pos_start.expand(2 * B, tc.model.first_l, -1) \
                     + lvl_pos[:, :tc.model.first_l]

    # Initialize latent representation.
    f_hat = sos.new_zeros(B, tc.model.Cvae, tc.model.patch_nums[-1], tc.model.patch_nums[-1])

    for b in tc.model.blocks:
        b.dense_blocks.attn.kv_caching(True)
        b.attn.kv_caching(True)
        b.ffn.forward_mode = forward_mode
        b.ffn.experts.forward_mode = 'triton_atomic'

    # Get all MoE modules within the blocks.
    moe_modules = [m for b in tc.model.blocks for m in b.modules() if hasattr(m, 'gate') and hasattr(m, 'router')]
    original_gates = {m: m.gate for m in moe_modules}

    cur_L = 0
    num_scales = len(tc.model.patch_nums)

    # Only construct the debug dictionary if requested.
    if save_sample:
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
    for si, pn in enumerate(tc.model.patch_nums):
        ratio = si / tc.model.num_stages_minus_1
        if save_sample:
            predicted_dictionary['ratio'].append(ratio)
        cur_L += pn * pn
        cond_BD_or_gss = tc.model.shared_ada_lin(cond_BD)
        if save_sample:
            predicted_dictionary['cond_BD_or_gss'].append(cond_BD_or_gss.detach().cpu().clone())

        x = next_token_map  # Autoregressive token input.
        list_of_outputs = []

        for index, b in enumerate(tc.model.blocks):                   
            if taus:
                b.ffn.tau = taus[si]
            else:
                b.ffn.tau = tau
            x, block_flops = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None, current_scale=si)
            total_flops += block_flops

        if save_sample:
            list_of_outputs.append(x.clone())

        logits_BlV = tc.model.get_logits(x, cond_BD, current_scale=si)

        if save_sample:
            predicted_dictionary['x'].append(x.detach().cpu().clone())
            predicted_dictionary['block_output'].append(list_of_outputs)
            
            usage_list_for_this_scale = []
            for idx in range(len(moe_modules)-1):
                moex = moe_modules[idx]
                try:
                    usage_list_for_this_scale.append(moex.routing_mask.clone())
                except:
                    usage_list_for_this_scale.append(torch.ones(
                        (x.shape[0], x.shape[1], moex.num_experts),
                        device=x.device,
                        dtype=x.dtype
                    ))
                    #x.shape[0] * x.shape[1] * self.ffn.num_experts * self.ffn.expert_dim
            predicted_dictionary['experts_per_token'].append(usage_list_for_this_scale)
        
        if save_routing_pattern and pn == tc.model.patch_nums[-1]:
            outfile = os.path.join("expert_routing.bin")
            masks = [m.routing_mask.detach() for m in moe_modules]
            stacked = torch.stack(masks, dim=0)
            # take the mean over the layer-axis
            expert_dist = stacked.mean(dim=0).cpu().numpy() 

            half = expert_dist.shape[0] // 2
            expert_dist = expert_dist[:half, :]       # shape now (128, 256)
            with open(outfile, "ab") as f:
                f.write(expert_dist.tobytes())
        
        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
        if save_sample:
            predicted_dictionary['logits_BlV'].append(logits_BlV.detach().cpu().clone())

        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        if save_sample:
            predicted_dictionary['idx_Bl'].append(idx_Bl.detach().cpu().clone())
       
        # --- Update latent representation ---        
        h_BChw = tc.model.vae_quant_proxy[0].embedding(idx_Bl)
        if save_sample:
            predicted_dictionary['h_BChw'].append(h_BChw.detach().cpu().clone())

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, tc.model.Cvae, pn, pn)

        f_hat, next_token_map = tc.model.vae_quant_proxy[0].get_next_autoregressive_input(si, num_scales, f_hat, h_BChw)
        if save_sample:
            predicted_dictionary['f_hat'].append(f_hat.clone())

        if save_sample:
            final_img = tc.model_vae.fhat_to_img(f_hat.clone())
            img = final_img[0].add_(1).mul_(0.5).permute(1, 2, 0).mul(255).clamp(0,255).cpu().numpy().astype(np.uint8)
            predicted_dictionary['img'].append(img)

        if si != tc.model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, tc.model.Cvae, -1).transpose(1, 2)
            next_token_map = tc.model.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + tc.model.patch_nums[si + 1] ** 2]     
            next_token_map = next_token_map.repeat(2, 1, 1)
            if save_sample:
                predicted_dictionary['next_token_map'].append(next_token_map.detach().cpu().clone())

    #print('-'*1000)
    for b in tc.model.blocks:
        b.dense_blocks.attn.kv_caching(False)
        b.attn.kv_caching(False)


    # Save the debug dictionary only if requested.
    if save_sample:
        os.makedirs(f'data', exist_ok=True)
        with open(f"data/{expert_index_switch}_with_tau_{tau}.pkl", "ab") as f:
            pickle.dump(predicted_dictionary, f)
            print(f"[DEBUG] Saved the data to data/{expert_index_switch}_with_tau_{tau}.pkl")

    # Restore original MoE gate functions.
    for m in moe_modules:
        m.gate = original_gates[m]

    return tc.model_vae.fhat_to_img(f_hat).add_(1).mul_(0.5), total_flops

