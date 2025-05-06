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

        print(new_state_dict.keys())
        pattern = re.compile(r"head_nm.ada_lin.*")
        # remove any key whose name mentions that class
        for k in list(new_state_dict.keys()):
            if pattern.match(k):
                print('found')
                del new_state_dict[k]

        model.load_state_dict(new_state_dict, strict=False)


        final_path = Path(args.path_file_moe)
        final_state = torch.load(final_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model, _ = replace_with_moes(model, **model_arg, module_filter_contition=dsti_mlp_filter_condition)
        model = model.to(args.device)

        print(state_dict.keys())
        pattern = re.compile(r"head_nm.ada_lin.*")
        # remove any key whose name mentions that class
        for k in list(state_dict.keys()):
            if pattern.match(k):
                print('found', k)
                del state_dict[k]


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

        cfg                = 4
        top_p              = 0.95 #0.96
        top_k              = 900
        more_smooth        = False
        final_flops = 0
        class_labels = (11,)  #@param {type:"raw"}
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=args.device)

        with torch.no_grad(), torch.autocast('cuda', enabled=True, dtype=torch.float16):
            recon_B3HW, total_flops = autoregressive_infer_cfg_with_expert_plot(
                tc=tc, B=B, label_B=label_B, cfg=cfg,
                top_k=top_k, top_p=top_p, rng=rng, more_smooth=more_smooth,
                tau=(1.0 if args.dsti_tau_as_list else tau),
                save_sample=True, forward_mode=forward_mode,
                taus=(tau if args.dsti_tau_as_list else False),
                expert_index_switch=args.expert_index_switch
            )
            final_flops += total_flops

  


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





