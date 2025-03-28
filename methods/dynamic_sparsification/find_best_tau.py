import logging
from datetime import datetime
from typing import Callable, Dict, List, Type

import torch
from omegaconf import OmegaConf
from torch import nn

from architectures.moe.moe_layers import ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moefication import add_routers, MoeficationMoE
from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import benchmark_moe, online_evaluate_moe, score_moe, autoregressive_infer_cfg_with_expert_plot, autoregressive_infer_cfg_test
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state, make_vae
from utils import load_model, save_state, remove_hooks, save_final, Mixup, get_lrs, \
    get_module_name, add_save_inputs_hook, add_save_output_norm_hook
from utils_var import arg_util
from utils_var.misc import create_npz_from_sample_folder
from trainer import VARTrainer
import dist
from architectures.moe.dsti import dsti_mlp_filter_condition
from architectures.moe.moefication import replace_with_moes
from architectures.pretrained import get_var_d16  
from pathlib import Path
from architectures.moe.dsti import replace_with_relu
from architectures.moe.dsti import find_gelu_activations
import copy
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import random
import torch, torchvision
import numpy as np
import os
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import shutil
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import autocast
import matplotlib.pyplot as plt
import torch_fidelity
import os
import re
import ast
import matplotlib.pyplot as plt
from PIL import Image
import time
from torch._C._profiler import ProfilerActivity
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import optuna
import torch
import random
import numpy as np
import os
import shutil
from PIL import Image
import optuna
import torch
from torch.utils.data import Subset
from collections import defaultdict
import optuna.visualization as vis
import plotly.graph_objects as go


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
    #_ , debug_data = make_image(tc.initial_model, args)

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
        model.load_state_dict(new_state_dict)

        final_path = Path(args.path_file_moe)
        final_state = torch.load(final_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model, _ = replace_with_moes(model, **model_arg, module_filter_contition=dsti_mlp_filter_condition)
        model = model.to(args.device)

        model.load_state_dict(state_dict)
        tc.moe_modules = add_routers(model, args.model_args)

    if args.use_router:
        final_router_path = Path('/home/jvincenti/D2DMoE/shared/results/effbench_runs/' + args.final_path_save + "_router" + "/final.pth")
        final_state = torch.load(final_router_path, map_location=args.device)
        state_dict = final_state['model_state']
        model_arg = final_state['args'].model_args
        model.load_state_dict(state_dict)
    
    tc.model = tc.accelerator.prepare(model)



def build_mini_dataset(folder_dataset, num_classes=1000):
    """
    folder_dataset: a torchvision.datasets.DatasetFolder or ImageFolder object
                    with .samples = [(filepath, class_index), ...].
    num_classes: how many classes you want to collect.
    
    Returns a Subset with exactly one sample per class (or until you reach num_classes).
    """
    seen_classes = set()
    selected_indices = []

    # folder_dataset.samples is a list of (filepath, class_index)
    for i, (path, class_idx) in enumerate(folder_dataset.samples):
        if class_idx not in seen_classes:
            seen_classes.add(class_idx)
            selected_indices.append(i)
            if len(seen_classes) == num_classes:
                break

    return Subset(folder_dataset, selected_indices)



def objective(trial, args, tc, mini_loader):
    """
    Evaluates a set of tau parameters over all batches in mini_loader.
    Returns a multi-objective tuple: (final_loss, total_experts).
    """

    # 1) Build tau_list: first 5 are constant 1.0; next 5 are non-increasing from [0.1, prev_tau].
    tau_list = [1.0] * 5
    prev_tau = 1.0
    for i in range(5):
        # Set lower bound: first iteration uses 0.9; others use -1.0
        lower_bound = prev_tau - 0.1
        # Suggest a float in the range [lower_bound, prev_tau]
        tau_val = trial.suggest_float(f"tau_{i}", lower_bound, prev_tau)
        tau_list.append(tau_val)
        # Update prev_tau to the new value for the next iteration
        prev_tau = tau_val
    
    # 2) Configure your model to use the chosen tau_list
    for b in tc.model.blocks:
        b.ffn.forward_mode = 'oracle'
        b.ffn.tau = tau_list
        b.ffn.experts.forward_mode = 'triton_atomic'
    
    # 3) Prepare to accumulate metrics across multiple batches
    all_losses = []
    all_experts = []

    # Pre-compute the patch-based scale weighting
    num_scales = len(args.patch_nums)
    if num_scales > 5:
        # E.g. first 5 scales get 0 weight, then linearly from 1.0 to 2.0
        high_scales = [1.0] * 5 #torch.linspace(1.0, 2.0, steps=num_scales - 5).tolist()
        scale_factors = [0.0] * 5 + high_scales
    else:
        scale_factors = [0.0] * num_scales

    weight_list = []
    for i, pn in enumerate(args.patch_nums):
        num_patches = pn * pn
        factor = scale_factors[i]
        weight_list.append(torch.full((num_patches,), factor, device=args.device))

    loss_weight = torch.cat(weight_list, dim=0)
    loss_weight = loss_weight / loss_weight.sum()  # normalize
    loss_weight = loss_weight.unsqueeze(0)         # (1, L)

    train_loss = nn.CrossEntropyLoss(reduction='none')

    moe_modules = [m for b in tc.model.blocks
                    for m in b.modules()
                    if hasattr(m, 'gate') and hasattr(m, 'router')]

    # 4) Loop over each batch in mini_loader
    for X, y in mini_loader:

        with torch.no_grad():
            B, V = y.shape[0], tc.model_vae.vocab_size
            X = X.to(dist.get_device(), non_blocking=True)
            label_B = y.to(dist.get_device(), non_blocking=True)
            gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X)  # shape is list of patch indices
            gt_BL = torch.cat(gt_idx_Bl, dim=1)                   # (B, L)
            x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            logits = tc.model(label_B, x_BLCv_wo_first_l)         # shape: (B, L, V)

        # Compute patch-based cross-entropy and apply weighting
        batch_task_loss = train_loss(
            logits.view(-1, V),
            gt_BL.view(-1)
        ).view(B, -1)  # => (B, L)
        
        batch_task_loss = (batch_task_loss * loss_weight).sum(dim=-1).mean()  
        # shape: (B, L) -> (B,) -> final single scalar for this batch

        # Measure "experts used" for this batch
        batch_experts = sum(moex.routing_mask.clone().sum() for moex in moe_modules)

        all_losses.append(batch_task_loss.item())
        all_experts.append(batch_experts)

    # 5) Average final metrics across all batches in mini_loader
    final_loss = float(sum(all_losses) / len(all_losses))
    final_experts = float(sum(all_experts) / len(all_experts))

    return (final_loss, final_experts/10000000)



def plot_all_trials_interactive(study):
    # Create an interactive Pareto front plot using Optuna's visualization API.
    fig = vis.plot_pareto_front(study)
    
    # Build custom hover text for each trial (showing the τ parameters)
    custom_texts = []
    for trial in study.trials:
        if trial.values is not None and trial.params:
            # Create a string with τ parameters, e.g., "tau_0=0.875, tau_1=0.840, ..."
            tau_text = ", ".join(
                f"{key}={trial.params[key]:.3f}" for key in trial.params if key.startswith("tau")
            )
            custom_texts.append(tau_text)
        else:
            custom_texts.append("")
    
    # Update the figure traces with the custom hover text.
    fig.update_traces(
        text=custom_texts,
        hovertemplate="%{text}<extra></extra>"
    )
    
    # Save the interactive plot as an HTML file.
    fig.write_html("Images/optuna_pareto_CE.html")
    del fig


def training_loop(args, tc):
    mini_dataset = build_mini_dataset(tc.train_loader.dataset, num_classes=128)
    mini_loader = torch.utils.data.DataLoader(mini_dataset, batch_size=args.batch_size, shuffle=False)
    # Multi-objective: minimize both loss and number of experts
    study = optuna.create_study(directions=["minimize", "minimize"])

    # We'll do 5 trials => 5 times we pick a new tau_list, run 1 batch, measure (loss, experts)
    study.optimize(lambda trial: objective(trial, args, tc, mini_loader), n_trials=2500)

    plot_all_trials_interactive(study)
    # After 5 single-batch runs, you have up to 5 solutions. 
    # Some might be "Pareto optimal".
    best_trials = study.best_trials
    print("Number of Pareto-optimal solutions:", len(best_trials))

    for i, t in enumerate(best_trials):
        print(f"Pareto solution {i}: values={t.values}, params={t.params}")



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


def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
