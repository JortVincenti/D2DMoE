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
import matplotlib.pyplot as plt



class RouterTrainingContext(TrainingContext):
    moe_modules: Dict[str, nn.Module] = None
    captured_layer_name_map: Dict[str, str] = None
    saved_inputs: Dict = None
    saved_output_norms: Dict = None
    hook_handles: List = None
    router_criterion_type: Type = None
    router_criterion: Callable = None
    initial_model = None



def setup_model(args, tc):
    assert args.model_class == 'dsti_router'

    # Base class
    model, tc.model_vae = get_var_d16()
    tc.initial_model = copy.deepcopy(model)
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


    
    for tau in args.dsti_tau_to_eval:
        print('tau', tau, 'for expert size', args.model_experts_size)
        tc.model.eval()

        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        tc.model.rng.manual_seed(seed)
        tc.initial_model.rng.manual_seed(seed)
        rng = tc.model.rng


        forward_mode = 'dynk_max'

        sample_folder  = f'../../../../scratch-shared/jvincenti/{args.final_path_save}_with_router_{args.use_router}_tau_{tau}_samples_256x256_for_expert_e{args.model_experts_size}'  # Where to save the 50,000 PNGs
        os.makedirs(sample_folder, exist_ok=True) 

        num_classes        = 1000               #  1000 ImageNet classes
        samples_per_class  = 10                 #  10k total
        all_labels = np.repeat(np.arange(num_classes), samples_per_class)   # (10 000,)

        # ---------------------------------------------------------------------------
        # Hyper-parameters
        # ---------------------------------------------------------------------------
        B = args.batch_size_eff           # e.g. 128
        cfg, top_p, top_k = 1.5, 0.96, 900
        more_smooth       = False
        forward_mode      = "dynk_max"

        # ---------------------------------------------------------------------------
        # Book-keeping
        # ---------------------------------------------------------------------------
        final_flops      = 0.0
        recon_samples    = {}             # class_idx -> np.uint8 image
        grid_saved       = False

        # ---------------------------------------------------------------------------
        # Iterate over all labels in strides of B *without* dropping the tail.
        # ---------------------------------------------------------------------------
        for start in range(0, len(all_labels), B):
            end          = min(start + B, len(all_labels))
            batch_labels = torch.as_tensor(all_labels[start:end], device="cuda")
            B_cur        = len(batch_labels)              # last batch may be smaller!

            with torch.no_grad(), torch.autocast("cuda", enabled=True, dtype=torch.float16):
                recon_B3HW, total_flops = autoregressive_infer_cfg_with_expert_plot(
                    tc=tc,
                    B=B_cur,                              # ← pass real batch size
                    label_B=batch_labels,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    rng=rng,
                    more_smooth=more_smooth,
                    tau=(1.0 if args.dsti_tau_as_list else tau),
                    save_sample=False,
                    forward_mode=forward_mode,
                    taus=(tau if args.dsti_tau_as_list else False),
                    expert_index_switch=args.expert_index_switch,
                    save_routing_pattern=True,
                )
                final_flops += total_flops

            # -----------------------------------------------------------------------
            # Save individual PNGs and collect first-of-class examples
            # -----------------------------------------------------------------------
            np_images = (recon_B3HW.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            for i, img_array in enumerate(np_images):
                cls         = batch_labels[i].item()
                global_idx  = start + i

                # 1) Save individual PNG
                Image.fromarray(img_array).save(
                    os.path.join(sample_folder, f"class_{cls:04d}_{global_idx:05d}.png")
                )

                # 2) Keep up to 25 unique class examples for the grid
                if not grid_saved and cls not in recon_samples and len(recon_samples) < 25:
                    recon_samples[cls] = img_array

            # 3) Once 25 classes collected, save the 5 × 5 grid (only once)
            if not grid_saved and len(recon_samples) == 25:
                rows, cols = 5, 5
                fig, axes  = plt.subplots(rows, cols, figsize=(10, 10))
                plt.subplots_adjust(wspace=0, hspace=0)

                for idx, cls in enumerate(sorted(recon_samples)[:25]):
                    r, c = divmod(idx, cols)
                    axes[r, c].imshow(recon_samples[cls], interpolation="nearest")
                    axes[r, c].axis("off")

                fig.savefig(
                    f"Images/{args.final_path_save}_with_router_{args.use_router}"
                    f"_tau_{tau}_scale_switch_{args.expert_index_switch}.png",
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close(fig)
                grid_saved = True

        # ---------------------------------------------------------------------------
        # Compute average flops *per* sample (all 10 000 samples, not just full batches)
        # ---------------------------------------------------------------------------
        final_flops /= len(all_labels) * 2

        # ---------------------------------------------------------------------------
        # FID / IS metrics
        # ---------------------------------------------------------------------------
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder,
            fid_statistics_file="adm_in256_stats.npz",
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )

        print("*" * 100)
        print(f"Final FID for τ = {tau}   ({args.final_path_save})")
        print(metrics_dict)
        print("Total FLOPs per sample:", final_flops)
        print("*" * 100)
        #shutil.rmtree(sample_folder)

  


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


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()





