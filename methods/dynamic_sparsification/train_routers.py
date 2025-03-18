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
    tc.initial_model = copy.deepcopy(model)
    #_ , debug_data = make_image(tc.initial_model, args)

    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.clone()


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
        #print('final_router_path', final_router_path)
        if final_router_path.exists():
            final_state = torch.load(final_router_path, map_location=args.device)
            state_dict = final_state['model_state']
            model_arg = final_state['args'].model_args
            model.load_state_dict(state_dict)

    #print('model', model)
    #print('final_router_path.exists()', final_router_path.exists())
    
    tc.model = tc.accelerator.prepare(model)


    if not args.fid and args.debug:
        for tau in args.dsti_tau_to_eval:
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

            # ------------------------------------------------------------------------------
            # 2. Configure output folder and sampling parameters.
            # ------------------------------------------------------------------------------
            type_of_model = (
                "MoE_FT_Gelu" if args.activation == "gelu" 
                else "MoE_FT_Relu" if args.activation == "relu" 
                else "MoE_no_FT"
            )
            #os.makedirs(sample_folder, exist_ok=True)
            # Check if directory exists
                
            cfg                = 4 #1.5
            more_smooth        = False

            class_labels = (1,)  #@param {type:"raw"}
            
            B = len(class_labels)
            label_B: torch.LongTensor = torch.tensor(class_labels, device=args.device)

            # Autoregressive sampling
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                    recon_B3HW, _, _, _ = autoregressive_infer_cfg_with_expert_plot(tc=tc, B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, rng=rng, more_smooth=more_smooth, tau=tau, debug_data=debug_data, compare_dicts=True, type_of_model=type_of_model, final_path_save=args.final_path_save)
        
    if args.fid:
        for tau in args.dsti_tau_to_eval:
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
            rng_2 = tc.initial_model.rng

            # ------------------------------------------------------------------------------
            # 2. Configure output folder and sampling parameters.
            # ------------------------------------------------------------------------------
            type_of_model = (
                "MoE_FT_Gelu" if args.activation == "gelu" 
                else "MoE_FT_Relu" if args.activation == "relu" 
                else "MoE_no_FT"
            )

            if args.use_router:
                forward_mode = 'dynk_max'
            else:
                forward_mode = 'oracle'

            sample_folder  = f'../../../../scratch-shared/jvincenti/{args.final_path_save}_with_router_{args.use_router}_tau_{tau}_samples_256x256'  # Where to save the 50,000 PNGs
            sample_folder_var = f'../../../../scratch-shared/jvincenti/base_var_tau_{tau}_samples_256x256'
            
            os.makedirs(sample_folder, exist_ok=True)
            if args.final_path_save =='base_data_moe' and tau == 1.0: 
                os.makedirs(sample_folder_var, exist_ok=True)
            # Check if directory exists
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
            B = 32  # e.g., B = 64

            num_total_samples = len(all_labels)
            num_batches = num_total_samples // B

            final_flops = 0
            final_mean_flops = 0
            batch_time = 0
            batch_time_base = 0

            recon_samples = {}      # dict mapping class_idx -> image (tensor or PIL)
            recon_var_samples = {}  # dict mapping class_idx -> image (tensor or PIL)

            for batch_idx in range(num_batches):
                # Create label_B for the batch: a combination of class indices.
                batch_labels = all_labels[batch_idx * B : (batch_idx + 1) * B]
                label_B = torch.tensor(batch_labels, device='cuda')
                
                # Autoregressive sampling with batch size B.
                with torch.no_grad():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):

                        recon_B3HW, mean_flops, total_flops, time_taken = autoregressive_infer_cfg_with_expert_plot(
                            tc=tc, B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p,
                            rng=rng, more_smooth=more_smooth, tau=tau, debug_data=None,
                            compare_dicts=False, type_of_model=type_of_model, final_path_save=None, forward_mode=forward_mode
                        )

                        batch_time += time_taken
                    
                        if args.final_path_save =='base_data_moe' and tau == 1.0:
                            torch.cuda.synchronize()
                            start_time = time.perf_counter()
                            recon_B3HW_var, _ = tc.initial_model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=seed, more_smooth=more_smooth, plotting_PCA=False, rng=rng_2)
                            # recon_B3HW_var, mean_flops, total_flops = autoregressive_infer_cfg_test(
                            # tc=tc, B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p,
                            # rng=rng, more_smooth=more_smooth, tau=tau, debug_data=None,
                            # compare_dicts=False, type_of_model=type_of_model, final_path_save=None
                            # )
                            end_time = time.perf_counter()
                            batch_time_base = batch_time_base + (end_time - start_time)


                        final_mean_flops += mean_flops
                        final_flops += total_flops

                    for i in range(B):
                        img_tensor = recon_B3HW[i].detach().cpu()
                        img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_array)
                        global_idx = batch_idx * B + i
                        filename = f"class_{batch_labels[i]:04d}_{global_idx:05d}.png"
                        img_pil.save(os.path.join(sample_folder, filename))

                        cls_label = batch_labels[i]
                        # Only store the first 25 unique classes
                        if cls_label not in recon_samples and len(recon_samples) < 25:
                            recon_samples[cls_label] = img_array

                    # 2) If we have exactly 25 stored images in recon_samples, produce a 5×5 grid
                    if len(recon_samples) == 25 and 1001 not in recon_samples:
                        chosen_moe_classes = sorted(recon_samples.keys())[:25]
                        num_moe = len(chosen_moe_classes)  # should be 25
                        rows_moe = 5
                        cols_moe = 5
                        fig_moe, axes_moe = plt.subplots(rows_moe, cols_moe, figsize=(10, 10))

                        plt.subplots_adjust(wspace=0, hspace=0)

                        for idx, cls_label in enumerate(chosen_moe_classes):
                            row = idx // cols_moe
                            col = idx % cols_moe
                            img_np = recon_samples[cls_label]  # shape [H, W, 3]
                            axes_moe[row, col].imshow(img_np, interpolation='nearest')
                            axes_moe[row, col].axis("off")

                        fig_moe.savefig(f"Images/{args.final_path_save}_with_router_{args.use_router}_tau_{tau}.png", bbox_inches="tight", pad_inches=0)
                        plt.close(fig_moe)
                        # Mark that we've already saved so we don't keep regenerating
                        recon_samples[1001] = 'end'

                    # 3) If final_path_save is base_data_moe AND tau == 1.0, we also handle recon_B3HW_var:
                    if args.final_path_save =='base_data_moe' and tau == 1.0:

                        for i in range(B):
                            img_tensor = recon_B3HW_var[i].detach().cpu()
                            img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            img_pil = Image.fromarray(img_array)
                            global_idx = batch_idx * B + i
                            filename = f"class_{batch_labels[i]:04d}_{global_idx:05d}.png"
                            img_pil.save(os.path.join(sample_folder_var, filename))

                            cls_label = batch_labels[i]
                            if cls_label not in recon_var_samples and len(recon_var_samples) < 25:
                                recon_var_samples[cls_label] = img_array

                        # Build a 5×5 grid for the first 25 classes in recon_var_samples
                        if len(recon_var_samples) == 25 and 1001 not in recon_var_samples:
                            chosen_var_classes = sorted(recon_var_samples.keys())[:25]
                            rows_var = 5
                            cols_var = 5
                            fig_var, axes_var = plt.subplots(rows_var, cols_var, figsize=(10, 10))
                            plt.subplots_adjust(wspace=0, hspace=0)

                            for idx, cls_label in enumerate(chosen_var_classes):
                                row = idx // cols_var
                                col = idx % cols_var
                                img_np = recon_var_samples[cls_label]
                                axes_var[row, col].imshow(img_np, interpolation='nearest')
                                axes_var[row, col].axis("off")

                            fig_var.savefig("Images/var_baseline_grid.png", bbox_inches="tight", pad_inches=0)
                            plt.close(fig_var)
                            recon_var_samples[1001] = 'end'

                            

            #print(f"Done! Images saved to: {sample_folder}")
            final_flops = final_flops/(num_batches*2*B)
            final_mean_flops = final_mean_flops/num_batches
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
            print(f'Final Fid for {tau}', args.final_path_save)
            print(metrics_dict)
            print('Total Flops per sample', final_flops)
            print('Average Flops per sample', final_mean_flops)
            print(f"Batch generation took {batch_time:.6f} seconds")
            print("*"*100)
            shutil.rmtree(sample_folder)


            if args.final_path_save =='base_data_moe' and tau == 1.0:
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=sample_folder_var,
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
                print(f'Final Fid for {tau} with base VAR')
                print(metrics_dict)
                print('Total Flops per sample', final_flops)
                print('Average Flops per sample', final_mean_flops)
                print('batch_time_base', batch_time_base)
                print("*"*100)
                shutil.rmtree(sample_folder_var)
        
    if args.debug:
        raise ValueError("args.debug is true therefore stopping here.")


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


def in_training_eval(args, tc):
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    set_for_eval_with_topk(tc, 1)
    cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader, tc)
    
    for tau in args.tau:
        set_for_eval_with_dynk(tc, tau, args.dsti_expert_selection_mode)
        cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader, tc)

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
        #in_training_eval(args, tc)
        tc.optimizer.zero_grad(set_to_none=True)
        set_for_train_iteration(tc) # Set forwardmore to 'all'
        # Account for gradient accumulation
        running_loss = 0

        X, y = next(train_iter)
        # forward
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
            # print('captured_output_norm:', captured_output_norm)
            # print('shape', captured_output_norm.shape)
            with torch.no_grad():
                # captured_output_norm size is (num_experts, batch_size * seq_len)
                router_label = captured_output_norm.view(captured_output_norm.size(0), input.size(0),
                                                            input.size(1))
                router_label = router_label.permute(1, 2, 0).detach()
            with tc.accelerator.autocast():
                router_output = router(input)
                router_loss = tc.router_criterion(router_output, router_label)
            # print('router_output:', router_output)
            # print('router_label:', router_label)
            # print(f'loss: {router_loss}')
            
            # Calculate the number of zeros and the total number of elements
            num_zeros_output = (router_output == 0).sum().item()
            total_elements_output = router_output.numel()
            percent_zeros_output = (num_zeros_output / total_elements_output) * 100

            num_zeros_label = (router_label == 0).sum().item()
            total_elements_label = router_label.numel()
            percent_zeros_label = (num_zeros_label / total_elements_label) * 100


            # print(f"Input stats: mean={input.mean().item()}, std={input.std().item()}, min={input.min().item()}, max={input.max().item()}")
            # print(f"Router output stats: mean={router_output.mean().item()}, std={router_output.std().item()}, min={router_output.min().item()}, max={router_output.max().item()}")
            # print(f"Router label stats: mean={router_label.mean().item()}, std={router_label.std().item()}, min={router_label.min().item()}, max={router_label.max().item()}")

            #print('-'*100)

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



    # # Now check each block's FFN experts:
    
    # for i, block in enumerate(tc.model.blocks):
    #     if hasattr(block, "ffn") and hasattr(block.ffn, "experts"):
    #         expert_weights = [expert.w.data for expert in block.ffn.experts.layers]
            
    #         # Check fc1 and fc2
    #         for jk in range(1, 3):
    #             ffn_key = f"blocks.{i}.ffn.fc{jk}.weight"
    #             if ffn_key not in initial_weights:
    #                 print(f"Block {i}: Key {ffn_key} not found in initial weights.")
    #                 continue

    #             original_ffn_weight = initial_weights[ffn_key]

    #             # 1) We'll just check the sum of all experts' weights
    #             #    to see if it's the same as the sum of the original layer's weight.
    #             # For example, sum up absolute or plain sums, or do a norm, etc.
    #             # We'll do a plain sum here.
    #             expert_sum = 0.0
    #             for w in expert_weights[jk - 1]:  # w => shape [whatever dims…]
    #                 expert_sum += w.sum().item()  # w is a 2D or 3D tensor, so .sum() is the sum of all elements.

    #             # Then compare with the original weight's sum:
    #             original_sum = original_ffn_weight.sum().item()

    #             sum_diff = abs(expert_sum - original_sum)
    #             if sum_diff < 1e-4:  # pick a threshold that’s good for your scale
    #                 print(f"Block {i} fc{jk}: The sum of all experts' weights ~ the sum of original FFN weights. sum_diff={sum_diff:.5f}")
    #             else:
    #                 print(f"Block {i} fc{jk}: The sum of experts' weights != original. sum_diff={sum_diff:.5f}")
    #                 unchanged = False

    #             # 2) Optionally, do more advanced checks (like L2 norm or a direct reorder).
    #             #    e.g., check L2 norm if you prefer:
    #             #    expert_l2, orig_l2 = 0.0, torch.norm(original_ffn_weight).item()
    #             #    for w in expert_weights[jk - 1]:
    #             #        expert_l2 += w.pow(2).sum().item()
    #             #    expert_l2 = math.sqrt(expert_l2)
    #             #    diff_l2 = abs(expert_l2 - orig_l2)
    #             #    # etc.


    #                 # set args



    
    # # Check non-FFN parameters normally:
    # for name, param in tc.model.named_parameters():
    #     if "ffn" not in name:
    #         if not torch.allclose(param, initial_weights[name], rtol=0, atol=0):
    #             print(f"Parameter {name} changed!")

        # for tau in args.dsti_tau_to_eval:
        #     seed = 0
        #     torch.manual_seed(seed)
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        #     tf32 = True
        #     torch.backends.cudnn.allow_tf32 = bool(tf32)
        #     torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        #     torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        #     tc.model.rng.manual_seed(seed)
        #     rng = tc.model.rng

        #     # ------------------------------------------------------------------------------
        #     # 2. Configure output folder and sampling parameters.
        #     # ------------------------------------------------------------------------------
        #     type_of_model = (
        #         "MoE_FT_Gelu" if args.activation == "gelu" 
        #         else "MoE_FT_Relu" if args.activation == "relu" 
        #         else "MoE_no_FT"
        #     )
        #     sample_folder  = f'../../../../scratch-shared/jvincent with base VARi}_tau_{tau}_samples_256x256'  # Where to save the 50,000 PNGs
            
        #     os.makedirs(sample_folder, exist_ok=True)
        #     # Check if directory exists
        #     num_classes        = 1000               # e.g. 1000 ImageNet classes
        #     samples_per_class  = 50                 # 50 images per class => 50k total
        #     cfg                = 1.5
        #     top_p              = 0.96
        #     top_k              = 900
        #     more_smooth        = False
                
        #     for class_idx in tqdm(range(num_classes), desc='Sampling'):
        #         # Create a batch of size = samples_per_class with the same class label
        #         label_B = torch.tensor([class_idx] * samples_per_class, device='cuda')
                
        #         # Autoregressive sampling
        #         with torch.inference_mode():
        #             with torch.autocast('cuda', enabled=True, dtype=torch.float16):
        #                 recon_B3HW = autoregressive_infer_cfg_with_expert_plot(tc=tc, B=samples_per_class, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, rng=rng, more_smooth=more_smooth, tau=tau, debug_data=debug_data, compare_dicts=False, type_of_model=type_of with base VAR_=None)
                
        #         # recon_B3HW should have shape [B, 3, H, W]. Save each image in the batch.
        #         for i in range(samples_per_class):
        #             # Convert each image [3,H,W] to a PIL Image (uint8)
        #             # Scale from [0,1] or [-1,1] as needed, depending on your model’s output
        #             # Here we assume recon_B3HW is in [0,1]. If it's in another scale,
        #             # adjust the multiplication and clamp accordingly.
        #             img_tensor = recon_B3HW[i].detach().cpu().clamp(0,1)
        #             img_pil = Image.fromarray(
        #                 (img_tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
        #             )
                    
        #             # Build filename like 00000001_123.png, indicating (class_####) + image index
        #             filename = f"class_{class_idx:04d}_{i:02d}.png"
        #             img_pil.save(os.path.join(sample_folder, filename))

        #     print(f"Done! Images saved to: {sample_folder}")

        #     # ------------------------------------------------------------------------------
        #     # 4. (Optional) Build the .npz file for FID/IS evaluation
        #     #    using your provided helper function:
        #     # ------------------------------------------------------------------------------
        #     # from utils.misc import create_npz_from_sample_folder
        #     npz_path = create_npz_from_sample_folder(sample_folder)
        #     print(f"Saved .npz file to {npz_path}")


            # # # Now check each block's FFN experts:
    
    # for i, block in enumerate(tc.model.blocks):
    #     if hasattr(block, "ffn") and hasattr(block.ffn, "experts"):
    #         expert_weights = [expert.w.data for expert in block.ffn.experts.layers]
            
    #         # Check fc1 and fc2
    #         for jk in range(1, 3):
    #             ffn_key = f"blocks.{i}.ffn.fc{jk}.weight"
    #             if ffn_key not in initial_weights:
    #                 print(f"Block {i}: Key {ffn_key} not found in initial weights.")
    #                 continue

    #             original_ffn_weight = initial_weights[ffn_key]

    #             # 1) We'll just check the sum of all experts' weights
    #             #    to see if it's the same as the sum of the original layer's weight.
    #             # For example, sum up absolute or plain sums, or do a norm, etc.
    #             # We'll do a plain sum here.
    #             expert_sum = 0.0
    #             for w in expert_weights[jk - 1]:  # w => shape [whatever dims…]
    #                 expert_sum += w.sum().item()  # w is a 2D or 3D tensor, so .sum() is the sum of all elements.

    #             # Then compare with the original weight's sum:
    #             original_sum = original_ffn_weight.sum().item()

    #             sum_diff = abs(expert_sum - original_sum)
    #             if sum_diff < 1e-4:  # pick a threshold that’s good for your scale
    #                 print(f"Block {i} fc{jk}: The sum of all experts' weights ~ the sum of original FFN weights. sum_diff={sum_diff:.5f}")
    #             else:
    #                 print(f"Block {i} fc{jk}: The sum of experts' weights != original. sum_diff={sum_diff:.5f}")
    #                 unchanged = False

    #             # 2) Optionally, do more advanced checks (like L2 norm or a direct reorder).
    #             #    e.g., check L2 norm if you prefer:
    #             #    expert_l2, orig_l2 = 0.0, torch.norm(original_ffn_weight).item()
    #             #    for w in expert_weights[jk - 1]:
    #             #        expert_l2 += w.pow(2).sum().item()
    #             #    expert_l2 = math.sqrt(expert_l2)
    #             #    diff_l2 = abs(expert_l2 - orig_l2)
    #             #    # etc.


    #                 # set args

    # Check non-FFN parameters normally:
    # if args.activation in ['relu']:
    #     for name, param in tc.model.named_parameters():
    #         if "ffn" not in name:
    #             if torch.allclose(param, initial_weights[name], rtol=0, atol=0):
    #                 raise ValueError("Params have not changed")
