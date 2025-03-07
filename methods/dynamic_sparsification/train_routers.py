import logging
from datetime import datetime
from typing import Callable, Dict, List, Type

import torch
from omegaconf import OmegaConf
from torch import nn

from architectures.moe.moe_layers import ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moefication import add_routers, MoeficationMoE
from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import benchmark_moe, online_evaluate_moe, score_moe, autoregressive_infer_cfg_with_expert_plot
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
import torch_fidelity


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
    class_labels = (0,)  #@param {type:"raw"}
    more_smooth = False # True for more smooth output


    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # sample

    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=args.device)

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float32, cache_enabled=True):    # using bfloat16 can be faster
            recon_B3HW, debug_data = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth, plotting_PCA=False)

    return recon_B3HW, debug_data

def setup_model(args, tc):
    assert args.model_class == 'dsti_router'

    # Base class
    model, tc.model_vae = get_var_d16()
    tc.initial_model = copy.deepcopy(model)
    #_ , debug_data = make_image(tc.initial_model, args)

    # initial_weights = {}
    # for name, param in model.named_parameters():
    #     initial_weights[name] = param.clone()


    # if args.activation in ['gelu', 'relu']:
    #     init_path = Path(args.path_file_ft)
    #     final_state = torch.load(init_path, map_location=args.device)
    #     state_dict = final_state['model_state']
    #     model_arg = final_state['args'].model_args

    #     if args.activation == 'relu':
    #         activations_to_sparsify = find_gelu_activations(model, **model_arg)
    #         model = replace_with_relu(model, activations_to_sparsify)

    #     model = model.to(args.device)
    #     new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    #     model.load_state_dict(new_state_dict)

    # final_path = Path(args.path_file_moe)
    # final_state = torch.load(final_path, map_location=args.device)
    # state_dict = final_state['model_state']
    # model_arg = final_state['args'].model_args
    # model, _ = replace_with_moes(model, **model_arg, module_filter_contition=dsti_mlp_filter_condition)
    # model = model.to(args.device)

    # model.load_state_dict(state_dict)
    # tc.moe_modules = add_routers(model, args.model_args)
    # init_fun = INIT_NAME_MAP[args.init_fun]
    # if init_fun is not None:
    #     init_fun(tc.model)

    # tc.model = tc.accelerator.prepare(model)


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


    # # Check non-FFN parameters normally:
    # for name, param in tc.model.named_parameters():
        # if "ffn" not in name:
        #     if not torch.allclose(param, initial_weights[name], rtol=0, atol=0):
        #         print(f"Parameter {name} changed!")

    if not args.fid:
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

            class_labels = (0,)  #@param {type:"raw"}
            
            B = len(class_labels)
            label_B: torch.LongTensor = torch.tensor(class_labels, device=args.device)

            # Autoregressive sampling
            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float32, cache_enabled=True):
                    recon_B3HW, _ = autoregressive_infer_cfg_with_expert_plot(tc=tc, B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, rng=rng, more_smooth=more_smooth, tau=tau, debug_data=debug_data, compare_dicts=True, type_of_model=type_of_model, final_path_save=args.final_path_save)
            

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
            model.rng.manual_seed(seed)
            rng = model.rng

            # ------------------------------------------------------------------------------
            # 2. Configure output folder and sampling parameters.
            # ------------------------------------------------------------------------------
            type_of_model = (
                "MoE_FT_Gelu" if args.activation == "gelu" 
                else "MoE_FT_Relu" if args.activation == "relu" 
                else "MoE_no_FT"
            )
            sample_folder  = f'../../../../scratch-shared/jvincenti/{args.final_path_save}_tau_{tau}_samples_256x256'  # Where to save the 50,000 PNGs
            
            os.makedirs(sample_folder, exist_ok=True)
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

            for batch_idx in range(num_batches):
                # Create label_B for the batch: a combination of class indices.
                batch_labels = all_labels[batch_idx * B : (batch_idx + 1) * B]
                label_B = torch.tensor(batch_labels, device='cuda')
                
                # Autoregressive sampling with batch size B.
                with torch.no_grad():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                        recon_B3HW, _ = model.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth, plotting_PCA=False)
                        # recon_B3HW, mean_flops, total_flops = autoregressive_infer_cfg_with_expert_plot(
                        #     tc=tc, B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95,
                        #     rng=rng, more_smooth=more_smooth, tau=tau, debug_data=debug_data,
                        #     compare_dicts=False, type_of_model=type_of_model, final_path_save=None
                        # )
                # final_mean_flops += mean_flops
                # final_flops += total_flops


                # Convert output to [0,1] range.
                recon_B3HW = (recon_B3HW + 1) / 2

                # Save each generated image.
                for i in range(B):
                    img_tensor = recon_B3HW[i].detach().cpu().clamp(0, 1)
                    img_array = (img_tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    img_pil = Image.fromarray(img_array)
                    # Use the class label from batch_labels to name the file, along with a global index.
                    global_idx = batch_idx * B + i
                    filename = f"class_{batch_labels[i]:04d}_{global_idx:05d}.png"
                    img_pil.save(os.path.join(sample_folder, filename))            
            #print(f"Done! Images saved to: {sample_folder}")
            # final_flops = final_flops/(num_classes*samples_per_class)
            # final_mean_flops = final_mean_flops/num_classes
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
            print("FID:", metrics_dict['frechet_inception_distance'])
            print("inception_score: ", metrics_dict['inception_score_mean'])
            #print('Avwerage Flops per sample', final_flops)
            print("*"*100)

            shutil.rmtree(sample_folder)
        
    asdasdas


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


def in_training_eval(args, tc, first_time=False):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        # set_for_eval_with_topk(tc, 1)
        # cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader, tc)
        for tau in args.dsti_tau_to_eval:
            # if tc.accelerator.is_main_process:
            #     logging.info(f'Testing on testset for tau={tau} on {args.eval_batches} batches.')
            # set_for_eval_with_dynk(tc, tau, args.dsti_expert_selection_mode)
            # cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader, tc)
            if first_time:
                #score_moe(unwrapped_model, tc.test_loader, tc, tau)


                # set args
                seed = 0 #@param {type:"number"}
                torch.manual_seed(seed)
                num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
                cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
                class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
                more_smooth = False # True for more smooth output


                # run faster
                tf32 = True
                torch.backends.cudnn.allow_tf32 = bool(tf32)
                torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
                torch.set_float32_matmul_precision('high' if tf32 else 'highest')

                # sample

                B = len(class_labels)
                label_B: torch.LongTensor = torch.tensor(class_labels, device='cuda')

                with torch.inference_mode():
                    with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                        autoregressive_infer_cfg_with_expert_plot(tc=tc, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth, tau=tau)


def training_loop(args, tc):
    if tc.accelerator.is_main_process:
        model_saved = datetime.now()
    train_iter = iter(tc.train_loader)
    count = 0
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    if args.mixup_alpha is not None or args.cutmix_alpha is not None: # Jort: This is not used
        mixup_mode = 'batch' if args.mixup_mode is None else args.mixup_mode
        mixup_smoothing = 0.1 if args.mixup_smoothing is None else args.mixup_smoothing
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, mode=mixup_mode,
            label_smoothing=mixup_smoothing, num_classes=unwrapped_model.number_of_classes)
    else:
        mixup_fn = None
    

    while tc.state.current_batch <= tc.last_batch:
        # save model conditionally
        if tc.accelerator.is_main_process:
            now = datetime.now()
            if (now - model_saved).total_seconds() > 60 * args.save_every:
                save_state(tc.accelerator, tc.state_path)
                model_saved = datetime.now()
        # model evaluation
        in_training_eval(args, tc, first_time=tc.state.current_batch == 0)
        tc.optimizer.zero_grad(set_to_none=True)
        set_for_train_iteration(tc) # Set forwardmore to 'all'
        # Account for gradient accumulation
        running_loss = 0
        for _ in range(args.gradient_accumulation_steps):
            try:
                X, y = next(train_iter)
            except StopIteration:
                train_iter = iter(tc.train_loader)
                X, y = next(train_iter)
            if mixup_fn is not None: # Jort: This is not used
                X, y = mixup_fn(X, y)
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
                if tc.accelerator.is_main_process:
                    tc.writer.add_scalar(f'Train/Router {moe_name} loss', router_loss.item(),
                                            global_step=tc.state.current_batch)
            loss = torch.stack(router_losses).mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # backward the loss
            tc.accelerator.backward(loss)
            running_loss += loss.item()
            # print('router_output:', router_output)
            # print('router_label:', router_label)
            print(f'loss: {loss}, router_losses: {router_losses}')
            print('-'*100)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Average loss', running_loss, global_step=tc.state.current_batch)
        if args.clip_grad_norm is not None:
            total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
            if tc.accelerator.is_main_process:
                tc.writer.add_scalar('Train/Gradient norm', total_norm.item(), global_step=tc.state.current_batch)

        # check if param is from router
        tc.optimizer.step()
        if tc.scheduler is not None:
            # log LRs
            if tc.accelerator.is_main_process:
                for i, lr in enumerate(get_lrs(tc.optimizer)):
                    tc.writer.add_scalar(f'Train/Group {i} LR', lr, global_step=tc.state.current_batch)
            if args.scheduler_class == 'reduce_on_plateau':
                tc.scheduler.step(loss)
            else:
                tc.scheduler.step()
        # bookkeeping
        tc.state.current_batch += 1


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

    # Log them with your TensorBoard or W&B writer
    tc.writer.add_scalar('Eval/Test loss', L_mean, global_step=tc.state.current_batch)
    tc.writer.add_scalar('Eval/Test accuracy', acc_mean, global_step=tc.state.current_batch)

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

    tc.writer.add_scalar('Eval/Model FLOPs', model_costs.total(), global_step=tc.state.current_batch)
    # If you have a key '' in model_params, you can log it:
    if '' in model_params:
        tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)
    else:
        # or the sum of all param counts if you prefer
        pass

    # --------------------------------------------------------
    # B) Optional MoE-based top-k / dyn-k evaluation
    # --------------------------------------------------------
    # If you do *not* need MoE logic, you can skip everything below or
    # wrap it in a condition `if tc.is_moe:` etc.

    # We typically only want to do the advanced MoE evaluation once,
    # e.g., if `tc.final_path` does not exist or if you specifically request it.
    if not tc.final_path.exists():

        # Possibly we want to save the current training state
        if tc.accelerator.is_main_process:
            save_state(tc.accelerator, tc.state_path)

        # If you rely on gating costs, you might retrieve them from somewhere:
        # cost_without_experts, token_expert_costs = ...
        # For demonstration, we'll assume you have them or skip them.

        final_losses = []
        final_scores = []
        final_flops = []
        final_expert_average_costs = []
        final_expert_utilization = []
        final_total_experts = []

        # Example: choose top-k or dyn-k approach
        if args.dsti_tau_to_eval is None and args.k_to_eval is None:
            raise ValueError('Must specify either dsti_tau_to_eval or k_to_eval for MoE final eval')

        if args.dsti_tau_to_eval is not None and args.k_to_eval is not None:
            raise ValueError('Cannot specify both tau and k for MoE final eval')

        if args.dsti_tau_to_eval is not None:
            # Evaluate for each tau
            for tau in args.dsti_tau_to_eval:
                if tc.accelerator.is_main_process:
                    logging.info(f'[MoE] Testing on testset for tau={tau}.')
                # set_for_eval_with_dynk(...) presumably changes forward_mode, sets tau, etc.
                set_for_eval_with_dynk(tc, tau, args.dsti_expert_selection_mode)

                # do your MoE evaluation. For example:
                (test_loss, test_acc, total_average_flops,
                 expert_avg_costs, executed_expert_tokens, total_expert_tokens) = online_evaluate_moe(
                     tc.accelerator,
                     tc.model,
                     tc.test_loader,
                     tc.criterion_type,
                     # cost_without_experts,
                     # token_expert_costs,
                     batches=args.test_batches,
                     return_counts=True
                )

                if tc.accelerator.is_main_process:
                    final_losses.append(test_loss)
                    final_scores.append(test_acc)
                    final_flops.append(total_average_flops)
                    final_expert_average_costs.append(expert_avg_costs)
                    final_expert_utilization.append(executed_expert_tokens)
                    final_total_experts.append(total_expert_tokens)

                    tc.writer.add_scalar(f'Eval MoE (tau={tau})/Test loss',
                                         test_loss, global_step=tc.state.current_batch)
                    tc.writer.add_scalar(f'Eval MoE (tau={tau})/Test accuracy',
                                         test_acc, global_step=tc.state.current_batch)
                    tc.writer.add_scalar(f'Eval MoE (tau={tau})/Model FLOPs',
                                         total_average_flops, global_step=tc.state.current_batch)

        if args.k_to_eval is not None:
            # Evaluate for each top-k
            for k_to_use in args.k_to_eval:
                if tc.accelerator.is_main_process:
                    logging.info(f'[MoE] Testing on testset for k={k_to_use}.')
                set_for_eval_with_topk(tc, k_to_use)  # sets forward_mode='topk', etc.

               
        # Save the MoE-specific results into final_results
        final_results['moe_test_losses'] = final_losses
        final_results['moe_test_scores'] = final_scores
        final_results['moe_test_flops'] = final_flops
        final_results['moe_expert_avg_costs'] = final_expert_average_costs
        final_results['moe_expert_utilization'] = final_expert_utilization
        final_results['moe_total_experts_used'] = final_total_experts
        # If tau-based, store the list of tau, else store the list of k
        final_results['hyperparam_values'] = args.dsti_tau_to_eval or args.k_to_eval

    # --------------------------------------------------------
    # C) Save the merged final results
    # --------------------------------------------------------
    if tc.accelerator.is_main_process:
        logging.info(f"Saving final results to {tc.final_path}")
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
    #setup_for_training(args, tc)
    #setup_data(args, tc)
    #setup_optimization(args, tc)
    #setup_state(tc)

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
        #     sample_folder  = f'../../../../scratch-shared/jvincenti/{args.final_path_save}_tau_{tau}_samples_256x256'  # Where to save the 50,000 PNGs
            
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
        #                 recon_B3HW = autoregressive_infer_cfg_with_expert_plot(tc=tc, B=samples_per_class, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, rng=rng, more_smooth=more_smooth, tau=tau, debug_data=debug_data, compare_dicts=False, type_of_model=type_of_model, final_path_save=None)
                
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