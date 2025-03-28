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

def inference_with_tau(tc, tau_list, num_classes=64):
    """
    Runs auto-regressive inference for each of `num_classes` (1 sample per class),
    returning a dict: label -> reconstruction.
    """
    # 1) Configure the model with tau_list
    for b in tc.model.blocks:
        b.ffn.forward_mode = 'oracle'
        b.ffn.tau = tau_list
        b.ffn.experts.forward_mode = 'triton_atomic'

    # 2) Possibly set seeds, etc. if needed for reproducibility:
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

    # 3) Build array of labels
    samples_per_class = 1
    all_labels = []
    for c in range(num_classes):
        all_labels.extend([c] * samples_per_class)
    all_labels = np.array(all_labels)

    B = 16
    num_total_samples = len(all_labels)
    num_batches = (num_total_samples + B - 1) // B

    # 4) For storing new reconstructions
    new_outputs = {cls: None for cls in range(num_classes)}
    full_flops = 0
    # 5) Loop over batches, run autoregressive_infer_cfg_with_expert_plot, etc.
    for batch_idx in range(num_batches):
        start_idx = batch_idx * B
        end_idx = min((batch_idx + 1) * B, num_total_samples)
        batch_labels = all_labels[start_idx:end_idx]
        label_B = torch.tensor(batch_labels, device='cuda')

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16, enabled=True):
                recon_B3HW, _, total_flops = autoregressive_infer_cfg_with_expert_plot(
                    tc=tc, B=label_B.shape[0], label_B=label_B,
                    cfg=1.5, top_k=900, top_p=0.96, rng=tc.model.rng,
                    more_smooth=False, tau=tau_list, debug_data=None,
                    compare_dicts=False, type_of_model=None,
                    final_path_save=None, forward_mode='oracle', taus=tau_list
                )
        full_flops += total_flops
        # recon_B3HW shape: (batch_size, C, H, W)
        for i in range(label_B.size(0)):
            cls = int(label_B[i].item())
            new_outputs[cls] = recon_B3HW[i].cpu()

    return new_outputs, full_flops/num_classes

def compare_reconstructions(new_outputs, reference_outputs):
    """
    Compute MSE for each class label between `new_outputs` and `reference_outputs`.
    Returns a dict mapping label -> MSE. If shapes differ or missing labels, MSE = None.
    """
    mse_dict = {}
    for label, ref_tensor in reference_outputs.items():
        new_tensor = new_outputs.get(label, None)
        if new_tensor is not None and ref_tensor.shape == new_tensor.shape:
            mse = torch.mean((ref_tensor - new_tensor) ** 2).item()
            mse_dict[label] = mse
        else:
            mse_dict[label] = None
    return mse_dict


def build_mini_dataset(tc, num_classes=64):
    """
    Creates a mini-dataset by picking 1 label per class (0..num_classes-1), then
    runs an autoregressive inference pass on each label (in batches). Returns
    a dictionary mapping each class label to its reconstructed output tensor.

    Args:
        folder_dataset: The dataset (e.g., ImageNet folder dataset), not directly used
                       for loading images in this snippet, but kept for context.
        tc: A context object holding the model (tc.model), among other parameters.
        num_classes: Number of classes to process (default=1000 for ImageNet).

    Returns:
        outputs_by_label (dict[int, torch.Tensor]):
            A dict where each key is a class label, and the value is the
            reconstruction tensor (e.g. shape [C, H, W]) for that label.
    """
    # --------------------------------------------------------------------------
    # 2) Define your fixed tau_list (e.g. all ones)
    # --------------------------------------------------------------------------
    tau_list = [1.0] * 10


    outputs_by_label, _ = inference_with_tau(
        tc=tc,
        tau_list=tau_list,
        num_classes=num_classes
    )
    
    return outputs_by_label

def objective(trial, args, tc, reference_outputs_by_label):
    """
    1. Suggest a new tau_list (non-increasing).
    2. Generate new reconstructions using that tau_list.
    3. Compare MSE with reference_outputs_by_label.
    4. Also measure 'experts' or 'flops' to use as the second objective.

    Returns: (avg_mse, avg_flops) for multi-objective.
    """
    # 1) Build tau_list
    tau_list = [1.0] * 5
    prev_tau = 1.0  # start at 1.0

    for i in range(5):
        # Set lower bound: first iteration uses 0.9; others use -1.0
        lower_bound = prev_tau - 0.1
        # Suggest a float in the range [lower_bound, prev_tau]
        tau_val = trial.suggest_float(f"tau_{i}", lower_bound, prev_tau)
        tau_list.append(tau_val)
        # Update prev_tau to the new value for the next iteration
        prev_tau = tau_val
    # 2) Inference with the new tau_list
    new_outputs, total_experts = inference_with_tau(tc, tau_list, num_classes=64)

    # 3) Compare with reference using MSE
    mse_dict = compare_reconstructions(new_outputs, reference_outputs_by_label)

    # Filter out None values and compute average
    valid_mses = [m for m in mse_dict.values() if m is not None]
    avg_mse = sum(valid_mses) / len(valid_mses)

    # 5) Return multi-objective
    return (avg_mse, total_experts/1000000000)


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
    fig.write_html("Images/optuna_pareto_MSE.html")
    del fig


def training_loop(args, tc):
    reference_outputs_by_label = build_mini_dataset(tc, num_classes=128)
    # Multi-objective: minimize both loss and number of experts
    study = optuna.create_study(directions=["minimize", "minimize"])

    # We'll do 5 trials => 5 times we pick a new tau_list, run 1 batch, measure (loss, experts)
    study.optimize(lambda trial: objective(trial, args, tc, reference_outputs_by_label), n_trials=2500)

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
