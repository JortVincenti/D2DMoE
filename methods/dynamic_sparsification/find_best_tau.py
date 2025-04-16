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


def build_mini_dataset(folder_dataset, num_classes=256):
    """
    Returns a Subset with up to `num_classes` distinct class indices
    from the original ImageFolder/FolderDataset.
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


def objective(param_dict, tc, mini_loader):
    """
    param_dict is a dictionary like: {
        "tau_0": <val>, "tau_1": <val>, ...
    }
    Returns a tuple: (final_loss, final_experts/1e7)
    """
    # 1) Gather the 5 tau_i from the dictionary
    tau_list = [1.0]*5  # Suppose first 5 positions are fixed=1.0
    for i in range(5):
        tau_val = param_dict[f"tau_{i}"]
        tau_list.append(tau_val)

    # 2) Configure your model using these tau values
    for b in tc.model.blocks:
        b.ffn.forward_mode = 'dynk_max'
        b.ffn.tau = tau_list
        b.ffn.experts.forward_mode = 'triton_atomic'
    
    # Prepare to accumulate metrics
    train_loss = nn.CrossEntropyLoss(reduction='none')
    moe_modules = [
        m for b in tc.model.blocks
        for m in b.modules()
        if hasattr(m, 'gate') and hasattr(m, 'router')
    ]

    all_losses = []
    all_experts = []

    # Set seeds, ensure deterministic operations, etc.
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

    # 3) Loop over the mini_loader to get average loss & average experts
    tc.model.eval()
    for X, y in mini_loader:
        B, V = y.shape[0], tc.model_vae.vocab_size
        X = X.to(dist.get_device(), non_blocking=True)
        label_B = y.to(dist.get_device(), non_blocking=True)

        gt_idx_Bl = tc.model_vae.img_to_idxBl(X)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)

        logits = tc.model(label_B, x_BLCv_wo_first_l)

        batch_task_loss = train_loss(
            logits.view(-1, V),
            gt_BL.view(-1)
        ).view(B, -1)

        batch_task_loss = batch_task_loss.mean()

        # measure "experts used" for this batch
        batch_experts = sum(moex.routing_mask.clone().sum() for moex in moe_modules)

        all_losses.append(batch_task_loss.item())
        all_experts.append(batch_experts)

    final_loss = float(sum(all_losses) / len(all_losses))
    final_experts = float(sum(all_experts) / len(all_experts))

    # 4) Return a tuple of objectives
    return (final_loss, final_experts)
    #return (final_loss, final_experts/1e7)


# -------------------------------------------------
# Helper to build valid tau combos for the grid
# -------------------------------------------------
def build_valid_tau_grid():
    """
    Generates valid combos with tau_0 >= tau_1 >= tau_2 >= tau_3 >= tau_4,
    each tau_i in [1.00 .. 0.70] stepping by 0.01
    """
    values = np.arange(1.0, 0.7, -0.02)
    candidate_values = [round(float(v), 2) for v in values]
    valid_grid = []
    valid_grid.append({
        "tau_0": 1.0,
        "tau_1": 1.0,
        "tau_2": 1.0,
        "tau_3": 1.0,
        "tau_4": 1.0
    })
    for t0 in candidate_values:
        for t1 in candidate_values:
            if t1 > t0:
                continue
            for t2 in candidate_values:
                if t2 > t1:
                    continue
                for t3 in candidate_values:
                    if t3 > t2:
                        continue
                    for t4 in candidate_values:
                        if t4 > t3:
                            continue
                        valid_grid.append({
                            "tau_0": t0,
                            "tau_1": t1,
                            "tau_2": t2,
                            "tau_3": t3,
                            "tau_4": t4
                        })


    return valid_grid


# -------------------------------------------------
# Compute 2D Pareto front for (loss, experts)
# -------------------------------------------------
def compute_pareto_front(results):
    """
    Given a list of dicts like:
      {
        'params': {...},
        'values': (loss, experts)
      }
    Return a sub-list of all non-dominated solutions in 'results'.
    
    A solution A is dominated by B if:
      B.loss <= A.loss AND B.experts <= A.experts
      (with at least one strict inequality)
    """
    pareto_solutions = []
    for rA in results:
        (lossA, expertsA) = rA['values']
        dominated = False
        for rB in results:
            (lossB, expertsB) = rB['values']
            if (lossB <= lossA and expertsB <= expertsA) and (lossB < lossA or expertsB < expertsA):
                # rA is dominated by rB
                dominated = True
                break
        if not dominated:
            pareto_solutions.append(rA)
    return pareto_solutions


# -------------------------------------------------
# Plot results in an interactive scatter
# -------------------------------------------------
def plot_all_results_interactive(results):
    """
    Creates an interactive Plotly scatter, showing each (loss, experts).
    We'll embed textual info about the tau parameters in the hover.
    """
    # Convert results to arrays for plotting
    xvals = []
    yvals = []
    hover_texts = []

    for res in results:
        (loss, experts) = res['values']
        xvals.append(loss)
        yvals.append(experts)
        param_dict = res['params']
        # Build a string like "tau_0=1.00, tau_1=0.95, ..."
        tau_text = ", ".join(
            f"{k}={param_dict[k]:.2f}"
            for k in sorted(param_dict.keys())
            if k.startswith("tau_")
        )
        hover_texts.append(tau_text)

    fig = go.Figure(
        data=go.Scatter(
            x=xvals,
            y=yvals,
            mode='markers',
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
        )
    )

    fig.update_layout(
        title="Pareto Scatter: (loss vs. experts)",
        xaxis_title="Loss",
        yaxis_title="Experts / 1e7"
    )
    fig.write_html("Images/pareto_CE.html")
    print("Wrote interactive plot to Images/pareto_CE.html.")


# -------------------------------------------------
# Main training loop with grid search
# -------------------------------------------------
def training_loop(args, tc):
    """
    Replaces the Optuna usage by manually iterating over a param grid.
    Then collects results, plots them, and reports Pareto-optimal solutions.
    """
    # 1) Build a small "mini" dataset for quick evaluation
    mini_dataset = build_mini_dataset(tc.train_loader.dataset, num_classes=16)
    mini_loader = torch.utils.data.DataLoader(mini_dataset, batch_size=16, shuffle=False)

    # 2) Generate only valid combos => no pruning needed
    valid_grid = build_valid_tau_grid()

    # 3) Evaluate each param combination
    results = []
    for param_dict in valid_grid:
        vals = objective(param_dict, tc, mini_loader)
        print(f'Run done for {param_dict} with results {vals}')
        results.append({
            'params': param_dict,
            'values': vals  # (loss, experts)
        })

    # 4) Produce the same style of interactive Pareto scatter
    #plot_all_results_interactive(results)

    # 5) Find and print the Pareto-optimal solutions
    pareto_solutions = compute_pareto_front(results)
    print("Number of Pareto-optimal solutions:", len(pareto_solutions))
    for i, sol in enumerate(pareto_solutions):
        print(f"Pareto solution {i}: values={sol['values']}, params={sol['params']}")

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
