#!/usr/bin/env python3
import os
import re
import ast
import math
import random
import logging
import shutil
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
import torch
import torchvision
from torch import nn, autocast
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from omegaconf import OmegaConf
import PIL.Image as PImage
import PIL.ImageDraw as PImageDraw
from PIL import Image
from tqdm import tqdm

import torch_fidelity

from architectures.moe.moe_layers import ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moefication import add_routers, MoeficationMoE, replace_with_moes
from architectures.moe.dsti import dsti_mlp_filter_condition, replace_with_relu, find_gelu_activations
from architectures.pretrained import get_var_d16

from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import autoregressive_infer_cfg_with_expert_plot
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, setup_state, make_vae
from utils import (load_model, save_state, remove_hooks, save_final,
                   Mixup, get_lrs, get_module_name,
                   add_save_inputs_hook, add_save_output_norm_hook)
from utils_var import arg_util
from utils_var.misc import create_npz_from_sample_folder
from trainer import VARTrainer
import dist
from collections import OrderedDict



class RouterTrainingContext(TrainingContext):
    moe_modules: Dict[str, nn.Module] = None
    captured_layer_name_map: Dict[str, str] = None
    saved_inputs: Dict = None
    saved_output_norms: Dict = None
    hook_handles: List = None
    router_criterion_type: Type = None
    router_criterion: Callable = None
    initial_model = None



def make_image(var, args):
    ############################# 2. Sample with classifier-free guidance
    # set args
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
    cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = (1,)  #@param {type:"raw"}
    more_smooth = False # True for more smooth output


    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    var.rng.manual_seed(seed)
    rng_2 = var.rng


    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # sample

    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=args.device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW, debug_data = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth, plotting_PCA=False, rng=rng_2)

    return recon_B3HW, debug_data

def setup_model(args, tc):
    assert args.model_class == 'dsti_router'

    # Base class
    model, tc.model_vae = get_var_d16()
    model.set_dense_module(scale_switch=args.expert_index_switch)

    if args.activation in ['gelu', 'relu']:
        init_path = Path(args.path_file_ft)
        final_state = torch.load(init_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args

        if args.activation == 'relu':
            activations_to_sparsify = find_gelu_activations(model, **model_arg)
            model = replace_with_relu(model, activations_to_sparsify)

        model = model.to(args.device)
        new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict, strict=False)


        final_path = Path(args.path_file_moe)
        final_state = torch.load(final_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model, _ = replace_with_moes(model, **model_arg, module_filter_contition=dsti_mlp_filter_condition)
        model = model.to(args.device)

        model.load_state_dict(state_dict, strict=False)
        tc.moe_modules = add_routers(model, args.model_args)

    if args.use_router:
        final_router_path = Path('/home/jvincenti/D2DMoE/shared/results/effbench_runs/' + args.final_path_save + "_router_e" + str(args.model_experts_size)  + "/final.pth")
        final_state = torch.load(final_router_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model.load_state_dict(state_dict, strict=False)

    
    tc.model = tc.accelerator.prepare(model)

def set_for_train_iteration(tc):
    tc.model.eval()
    tc.model.requires_grad_(False)
    for moe_name, moe_module in tc.moe_modules.items():
        assert moe_module.router is not None
        moe_module.router.train()
        moe_module.router.requires_grad_(True)
        moe_module.forward_mode = 'all'


def get_captured_layer_name_map(model, moe_modules: Dict, layer: str):
    module_names_map = {}
    for moe_name, moe_module in moe_modules.items():
        # TODO support other experts implementations than ExecuteAllExperts
        if isinstance(moe_module.experts, ExecuteAllExperts):
            # see ExecuteAllExperts class
            if layer == 'intermediate':
                module_names_map[moe_name] = get_module_name(model, moe_module.experts.layers[0])
            elif layer == 'output':
                module_names_map[moe_name] = get_module_name(model, moe_module.experts.layers[1])
            else:
                raise ValueError(f'Unknown layer for labels construction: {layer}')
        elif isinstance(moe_module.experts, CustomKernelExperts):
            if layer == 'intermediate':
                module_names_map[moe_name] = get_module_name(model,
                                                             moe_module.experts.intermediate_activation_extraction_stub)
            elif layer == 'output':
                module_names_map[moe_name] = get_module_name(model,
                                                             moe_module.experts.output_activation_extraction_stub)
            else:
                raise ValueError(f'Unknown layer for labels construction: {layer}')
    return module_names_map


def setup_for_training(args, tc):
    set_for_train_iteration(tc)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    tc.captured_layer_name_map = get_captured_layer_name_map(unwrapped_model, tc.moe_modules,
                                                             args.dsti_router_labels_layer)
    tc.saved_inputs, input_handles = add_save_inputs_hook(unwrapped_model, tc.moe_modules.keys())
    tc.saved_output_norms, output_handles = add_save_output_norm_hook(unwrapped_model,
                                                                      tc.captured_layer_name_map.values(),
                                                                      ord=args.dsti_router_labels_norm,
                                                                      dim=-1)                                                 
    tc.hook_handles = input_handles + output_handles
    tc.router_criterion_type = LOSS_NAME_MAP[args.router_loss_type]
    criterion_args = args.router_loss_args if args.router_loss_args is not None else {}
    tc.router_criterion = tc.router_criterion_type(reduction='mean', **criterion_args)


def set_for_eval_with_dynk(tc, tau, mode=None):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = 'dynk_max' if mode is None else mode
            m.tau = tau


def set_for_eval_with_topk(tc, k):
    tc.model.eval()
    for m in tc.model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = 'topk'
            m.k = k

def training_loop(args, tc):
    model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    current_batch = 0

    # for _ in range(current_batch - 1):
    #     next(train_iter)

    while current_batch <= tc.last_batch:
        # save model conditionally
        now = datetime.now()
        if (now - model_saved).total_seconds() > 60*60:
            if isinstance(args.runs_dir, str):
                args.runs_dir = Path("runs")
            args.runs_dir.mkdir(parents=True, exist_ok=True)
            run_name = str(args.final_path_save) + "_" + "router"
            run_dir = args.runs_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            tc.final_path = run_dir / 'final.pth'

            model_saved = datetime.now()
            final_results = {}
            final_results['args'] = args            
            final_results['model_state'] = tc.model.state_dict()

            save_final(args, tc.final_path, final_results)

      
        # model evaluation
        tc.optimizer.zero_grad(set_to_none=True)
        set_for_train_iteration(tc) # Set forwardmore to 'all'
        # Account for gradient accumulation
        running_loss = 0

        X, y = next(train_iter)
   
        with torch.no_grad():    
            B, V = y.shape[0], tc.model_vae.vocab_size
            X = X.to(dist.get_device(), non_blocking=True)
            label_B = y.to(dist.get_device(), non_blocking=True)
            gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X) 
            x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            tc.model(label_B, x_BLCv_wo_first_l)


        router_losses = []
        for moe_name, moe in tc.moe_modules.items():
            router = moe.router
            input = tc.saved_inputs[moe_name][0]
            captured_output_norm = tc.saved_output_norms[tc.captured_layer_name_map[moe_name]]

            with torch.no_grad():
                # captured_output_norm size is (num_experts, batch_size * seq_len)
                router_label = captured_output_norm.view(captured_output_norm.size(0), input.size(0),
                                                            input.size(1))
                router_label = router_label.permute(1, 2, 0).detach()
            with tc.accelerator.autocast():
                router_output = router(input)

                router_loss = tc.router_criterion(router_output, router_label)


            router_losses.append(router_loss)
        loss = torch.stack(router_losses).mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        # backward the loss
        tc.accelerator.backward(loss)
        running_loss += loss.item()
        # print('router_output:', router_output)
        # print('router_label:', router_label)
        print(f'loss: {loss}')

        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)

        # check if param is from router
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        # bookkeeping
        current_batch += 1


def final_eval(args, tc):
    """
    A single function that runs:
      1) The standard evaluation from `eval_ep` in VARTrainer (loss/accuracy/flops).
      2) The optional MoE-based evaluation (top-k or dyn-k) if `dsti_tau_to_eval` or `k_to_eval` is set.
      3) Logs and saves everything to `tc.final_path`.
    """

    # --------------------------------------------------------
    # A) Standard VARTrainer evaluation
    # --------------------------------------------------------
    L_mean, L_tail, acc_mean, acc_tail, tot, duration, model_costs, model_params = tc.trainer.eval_ep(tc.val_loader)

    # Print out standard stats
    print(f"Final evaluation (VARTrainer) completed:\n"
          f"  Mean Loss: {L_mean:.4f}, Tail Loss: {L_tail:.4f}\n"
          f"  Mean Accuracy: {acc_mean:.2f}%, Tail Accuracy: {acc_tail:.2f}%\n"
          f"  Total samples: {tot}, Duration: {duration:.2f}s")


    # We unwrap the model for checkpointing or direct access
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)

    # Build a results dictionary to store this portion
    final_results = {
        'args': args,
        'model_state': unwrapped_model.state_dict(),
        'final_score': acc_mean,
        'final_loss': L_mean,
        'model_flops': model_costs.total(),
        'model_flops_by_module': dict(model_costs.by_module()),
        'model_flops_by_operator': dict(model_costs.by_operator()),
        'model_params': dict(model_params),
    }

    save_final(args, tc.final_path, final_results)


def train(args):
    logging.basicConfig(
        format=(
            '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] ' '%(message)s'
        ),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    args: arg_util.Args = arg_util.init_dist_and_get_args(args)
    logging.info('Configured logging')
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_for_training(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)

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


    training_loop(args, tc)
    final_eval(args, tc)


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()





  # for i in range(B):
                    #     img_tensor = recon_B3HW[i].detach().cpu()
                    #     img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    #     img_pil = Image.fromarray(img_array)
                    #     global_idx = batch_idx * B + i
                    #     filename = f"class_{batch_labels[i]:04d}_{global_idx:05d}.png"
                    #     img_pil.save(os.path.join(sample_folder, filename))

                    #     cls_label = batch_labels[i]
                    #     # Only store the first 25 unique classes
                    #     if cls_label not in recon_samples and len(recon_samples) < 25:
                    #         recon_samples[cls_label] = img_array

                    # # 2) If we have exactly 25 stored images in recon_samples, produce a 5×5 grid
                    # if len(recon_samples) == 25 and 1001 not in recon_samples:
                    #     chosen_moe_classes = sorted(recon_samples.keys())[:25]
                    #     num_moe = len(chosen_moe_classes)  # should be 25
                    #     rows_moe = 5
                    #     cols_moe = 5
                    #     fig_moe, axes_moe = plt.subplots(rows_moe, cols_moe, figsize=(10, 10))

                    #     plt.subplots_adjust(wspace=0, hspace=0)

                    #     for idx, cls_label in enumerate(chosen_moe_classes):
                    #         row = idx // cols_moe
                    #         col = idx % cols_moe
                    #         img_np = recon_samples[cls_label]  # shape [H, W, 3]
                    #         axes_moe[row, col].imshow(img_np, interpolation='nearest')
                    #         axes_moe[row, col].axis("off")

                    #     fig_moe.savefig(f"Images/{args.final_path_save}_with_router_{args.use_router}_tau_{tau}.png", bbox_inches="tight", pad_inches=0)
                    #     plt.close(fig_moe)
                    #     # Mark that we've already saved so we don't keep regenerating
                    #     recon_samples[1001] = 'end'

                    # 3) If final_path_save is base_data_moe AND tau == 1.0, we also handle recon_B3HW_var:
                    # if args.final_path_save =='base_data_moe' and tau == 1.0:

                    #         for i in range(B):
                    #             img_tensor = recon_B3HW_var[i].detach().cpu()
                    #             img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    #             img_pil = Image.fromarray(img_array)
                    #             global_idx = batch_idx * B + i
                    #             filename = f"class_{batch_labels[i]:04d}_{global_idx:05d}.png"
                    #             img_pil.save(os.path.join(sample_folder_var, filename))

                    #             cls_label = batch_labels[i]
                    #             if cls_label not in recon_var_samples and len(recon_var_samples) < 25:
                    #                 recon_var_samples[cls_label] = img_array

                    #         # Build a 5×5 grid for the first 25 classes in recon_var_samples
                    #         if len(recon_var_samples) == 25 and 1001 not in recon_var_samples:
                    #             chosen_var_classes = sorted(recon_var_samples.keys())[:25]
                    #             rows_var = 5
                    #             cols_var = 5
                    #             fig_var, axes_var = plt.subplots(rows_var, cols_var, figsize=(10, 10))
                    #             plt.subplots_adjust(wspace=0, hspace=0)

                    #             for idx, cls_label in enumerate(chosen_var_classes):
                    #                 row = idx // cols_var
                    #                 col = idx % cols_var
                    #                 img_np = recon_var_samples[cls_label]
                    #                 axes_var[row, col].imshow(img_np, interpolation='nearest')
                    #                 axes_var[row, col].axis("off")

                    #             fig_var.savefig("Images/var_baseline_grid.png", bbox_inches="tight", pad_inches=0)
                    #             plt.close(fig_var)
                    #             recon_var_samples[1001] = 'end'



