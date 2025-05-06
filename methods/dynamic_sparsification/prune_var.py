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
    # Base class
    for pct_remove in args.pruning_percentage:
        model, tc.model_vae = get_var_d16()
        tc.model = tc.accelerator.prepare(model)
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
        rng = tc.model.rng


        # if args.use_router:
        #     forward_mode = 'dynk_max' #'centroids'
        # else:
        #     forward_mode = 'oracle' #'centroids' 
        forward_mode = 'dynk_max'

        sample_folder  = f'../../../../scratch-shared/jvincenti/pruned_var'  # Where to save the 50,000 PNGs
        os.makedirs(sample_folder, exist_ok=True) 

        num_classes        = 1000               #  1000 ImageNet classes
        samples_per_class  = 10                 #  10k total
        cfg                = 1.5
        top_p              = 0.96
        top_k              = 900
        more_smooth        = False


        # Assume num_classes and samples_per_class are defined.
        # Create a list of labels: each class is repeated samples_per_class times.
        all_labels = []
        for class_idx in range(num_classes):
            all_labels.extend([class_idx] * samples_per_class)
        all_labels = np.array(all_labels)

        # Define your new batch size B (which can be greater than samples_per_class)
        B = args.batch_size_eff  # e.g., B = 128

        num_total_samples = len(all_labels)
        num_batches = num_total_samples // B
        warmup_batches = num_batches // 2

        final_flops = 0
        final_mean_flops = 0
        batch_time = 0
        batch_time_base = 0

        recon_samples = {}       # class_idx -> img array
        grid_saved = False       # make sure we only save the grid once

        for batch_idx in range(num_batches):
            # prepare labels and sample
            batch_labels = all_labels[batch_idx * B : (batch_idx + 1) * B]
            label_B = torch.tensor(batch_labels, device='cuda')

            with torch.no_grad(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
                recon_B3HW = model.autoregressive_infer_cfg_pruning(
                    B=B, label_B=label_B,
                    g_seed=None, cfg=cfg, top_k=top_k, top_p=top_p,
                    more_smooth=False, rng=rng, prune_from_stage=args.expert_index_switch, pct_remove=pct_remove
                )

            # move to CPU and convert to numpy uint8 arrays
            batch_images = recon_B3HW.detach().cpu()                   # (B, C, H, W)
            np_images = (batch_images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

            # save each sample and collect first-of-class
            for i, img_array in enumerate(np_images):
                cls = batch_labels[i]
                global_idx = batch_idx * B + i

                # 1) Save individual file
                img_pil = Image.fromarray(img_array)
                fname = f"class_{cls:04d}_{global_idx:05d}.png"
                img_pil.save(os.path.join(sample_folder, fname))


        # Now calculate_metrics:
        input2 = None
        fid_statistics_file = 'adm_in256_stats.npz'

        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        print("*"*100)
        print(f'Results for Pruned Var with {pct_remove}% after scale {args.expert_index_switch}')
        print(metrics_dict)
        print("*"*100)
        shutil.rmtree(sample_folder)

    


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





