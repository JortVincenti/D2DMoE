import logging
from datetime import datetime
from typing import Callable, Dict, List, Type

import torch
from omegaconf import OmegaConf
from torch import nn

from architectures.moe.moe_layers import ExecuteAllExperts, CustomKernelExperts
from architectures.moe.moefication import add_routers, MoeficationMoE
from common import get_default_args, INIT_NAME_MAP, LOSS_NAME_MAP
from eval import benchmark_moe, online_evaluate_moe
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state, make_vae, final_eval
from utils import load_model, save_state, remove_hooks, save_final, Mixup, get_lrs, \
    get_module_name, add_save_inputs_hook, add_save_output_norm_hook
from utils_var import arg_util
from trainer import VARTrainer
import dist


class RouterTrainingContext(TrainingContext):
    moe_modules: Dict[str, nn.Module] = None
    captured_layer_name_map: Dict[str, str] = None
    saved_inputs: Dict = None
    saved_output_norms: Dict = None
    hook_handles: List = None
    router_criterion_type: Type = None
    router_criterion: Callable = None


def setup_model(args, tc):
    assert args.model_class == 'dsti_router'
    model, base_args, _, tc.model_var_wo_ddp = load_model(args, args.base_on, args.exp_id)
    tc.moe_modules = add_routers(model, args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(tc.model)
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


def in_training_eval(args, tc):
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        unwrapped_model = tc.accelerator.unwrap_model(tc.model)
        set_for_eval_with_topk(tc, 1)
        #cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader)
        for tau in args.dsti_tau_to_eval:
            if tc.accelerator.is_main_process:
                logging.info(f'Testing on testset for tau={tau} on {args.eval_batches} batches.')
            set_for_eval_with_dynk(tc, tau, args.dsti_expert_selection_mode)
            # test_loss, test_acc, test_average_flops, _ = online_evaluate_moe(tc.accelerator,
            #                                                                  tc.model,
            #                                                                  tc.test_loader,
            #                                                                  tc.criterion_type,
            #                                                                  cost_without_experts,
            #                                                                  token_expert_costs,
            #                                                                  batches=args.eval_batches)
            # if tc.accelerator.is_main_process:
            #     logging.info(f'Testing on trainset for tau={tau} on {args.eval_batches} batches.')
            # train_loss, train_acc, train_average_flops, _ = online_evaluate_moe(tc.accelerator,
            #                                                                     tc.model,
            #                                                                     tc.train_loader,
            #                                                                     tc.criterion_type,
            #                                                                     cost_without_experts,
            #                                                                     token_expert_costs,
            #                                                                     batches=args.eval_batches)
            # if tc.accelerator.is_main_process:
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Test loss', test_loss,
            #                          global_step=tc.state.current_batch)
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Test accuracy', test_acc,
            #                          global_step=tc.state.current_batch)
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Test model FLOPs', test_average_flops,
            #                          global_step=tc.state.current_batch)
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Train loss', train_loss,
            #                          global_step=tc.state.current_batch)
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Train accuracy', train_acc,
            #                          global_step=tc.state.current_batch)
            #     tc.writer.add_scalar(f'Eval with tau={tau}/Train model FLOPs', train_average_flops,
            #                          global_step=tc.state.current_batch)


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
    if args.dsti_separate_router_training: # Jort This is not used
        org_module_names = list(tc.moe_modules.keys())

        if tc.state.current_batch != 0:
            raise ValueError('Restarting from non-zero batch is not supported for separate router training.')
        for module_name in org_module_names:
            remove_hooks(tc.hook_handles)
            tc.saved_inputs, input_handles = add_save_inputs_hook(unwrapped_model, [module_name])
            tc.saved_output_norms, output_handles = add_save_output_norm_hook(unwrapped_model,
                                                                              [module_name],
                                                                              ord=args.dsti_router_labels_norm,
                                                                              dim=-1)
            tc.hook_handles = input_handles + output_handles
            tc.state.current_batch = 0
            while tc.state.current_batch <= tc.last_batch:
                # save model conditionally
                if tc.accelerator.is_main_process:
                    now = datetime.now()
                    if (now - model_saved).total_seconds() > 60 * args.save_every:
                        save_state(tc.accelerator, tc.state_path)
                        model_saved = datetime.now()
                # model evaluation
                in_training_eval(args, tc)
                tc.optimizer.zero_grad(set_to_none=True)
                set_for_train_iteration(tc)
                # Account for gradient accumulation
                running_loss = 0
                for _ in range(args.gradient_accumulation_steps):
                    # batch preparation
                    try:
                        X, y = next(train_iter)
                    except StopIteration:
                        train_iter = iter(tc.train_loader)
                        X, y = next(train_iter)
                    if mixup_fn is not None:
                        X, y = mixup_fn(X, y)
                    # forward
                    with torch.no_grad():
                        _ = tc.model(X)
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
                        if tc.accelerator.is_main_process:
                            tc.writer.add_scalar(f'Train/Router {moe_name} loss', router_loss.item(),
                                                 global_step=tc.state.current_batch)
                    loss = torch.stack(router_losses).mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    # backward the loss
                    tc.accelerator.backward(loss)
                    
                    running_loss += loss.item()
                if tc.accelerator.is_main_process:
                    tc.writer.add_scalar('Train/Average loss', running_loss, global_step=tc.state.current_batch)
                if args.clip_grad_norm is not None:
                    total_norm = tc.accelerator.clip_grad_norm_(tc.model.parameters(), args.clip_grad_norm)
                    if tc.accelerator.is_main_process:
                        tc.writer.add_scalar('Train/Gradient norm', total_norm.item(),
                                             global_step=tc.state.current_batch)
                # gradient step
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
            tc.hook_handles = []
    else: 
        while tc.state.current_batch <= tc.last_batch:
            # save model conditionally
            if tc.accelerator.is_main_process:
                now = datetime.now()
                if (now - model_saved).total_seconds() > 60 * args.save_every:
                    save_state(tc.accelerator, tc.state_path)
                    model_saved = datetime.now()
            # model evaluation
            #in_training_eval(args, tc)
            tc.optimizer.zero_grad(set_to_none=True)
            set_for_train_iteration(tc)
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
                    gt_idx_Bl: List[ITen] = tc.model_vae.img_to_idxBl(X) # This does not return None
                    x_BLCv_wo_first_l: Ten = tc.model_vae.quantize.idxBl_to_var_input(gt_idx_Bl)
                    tc.model(label_B, x_BLCv_wo_first_l)

                router_losses = []
                for moe_name, moe in tc.moe_modules.items():
                    # print('tc.captured_layer_name_map[moe_name]', tc.captured_layer_name_map[moe_name])
                    router = moe.router
                    # print('router:', router)
                    # print("*"*100)
                    # print('-'*100)
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
            # print("INFO: Printing grad info for all modules")
            # for moe_name, moe in tc.moe_modules.items():
            #     print(f"=== Gradient info for {moe_name} ===")
            #     router = moe.router
            #     for param_name, param in router.named_parameters():
            #         if param.requires_grad:
            #             if param.grad is None:
            #                 print(f"  {param_name}: grad is None (no gradient received)")
            #             else:
            #                 # Print the gradient norm or any stat you want:
            #                 print(f"  {param_name}: grad norm = {param.grad.norm().item():.6f}")
            #         else:
            #             print(f"  {param_name}: requires_grad=False (frozen parameter)")

            # if count == 10:
            #     sfsfsdfs
            # count += 1

# def final_eval(args, tc):
#     if not tc.final_path.exists():
#         if tc.accelerator.is_main_process:
#             save_state(tc.accelerator, tc.state_path)
#         if hasattr(tc, 'hook_handles'):
#             remove_hooks(tc.hook_handles)
#         set_for_eval_with_topk(tc, 1)
#         unwrapped_model = tc.accelerator.unwrap_model(tc.model)
#         del tc.optimizer
#         #cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, tc.test_loader)
#         final_scores = []
#         final_losses = []
#         final_flops = []
#         final_expert_average_costs = []
#         final_expert_utilization = []
#         final_total_experts = []
#         if args.dsti_tau_to_eval is None and args.k_to_eval is None:
#             raise ValueError('Must specify either dsti_tau_to_eval or dsti_k_to_eval')
#         if args.dsti_tau_to_eval is not None and args.k_to_eval is not None:
#             raise ValueError('Must specify either dsti_tau_to_eval or dsti_k_to_eval, not both')
#         if args.dsti_tau_to_eval is not None:
#             for tau in args.dsti_tau_to_eval:
#                 if tc.accelerator.is_main_process:
#                     logging.info(f'Testing on testset for tau={tau}.')
#                 set_for_eval_with_dynk(tc, tau, args.dsti_expert_selection_mode)
#                 test_loss, test_acc, total_average_flops, expert_average_costs, executed_expert_tokens, total_expert_tokens = online_evaluate_moe(
#                     tc.accelerator,
#                     tc.model,
#                     tc.test_loader,
#                     tc.criterion_type,
#                     cost_without_experts,
#                     token_expert_costs,
#                     batches=args.test_batches,
#                     return_counts=True)
#                 if tc.accelerator.is_main_process:
#                     final_losses.append(test_loss)
#                     final_scores.append(test_acc)
#                     # benchmark model efficiency
#                     final_flops.append(total_average_flops)
#                     final_expert_average_costs.append(expert_average_costs)
#                     final_expert_utilization.append(executed_expert_tokens)
#                     final_total_experts.append(total_expert_tokens)
#                     # log
#                     tc.writer.add_scalar(f'Eval with tau={tau}/Test loss', test_loss,
#                                          global_step=tc.state.current_batch)
#                     tc.writer.add_scalar(f'Eval with tau={tau}/Test accuracy', test_acc,
#                                          global_step=tc.state.current_batch)
#                     tc.writer.add_scalar(f'Eval with tau={tau}/Model FLOPs', total_average_flops,
#                                          global_step=tc.state.current_batch)
#         if args.k_to_eval is not None:
#             for k_to_use in args.k_to_eval:
#                 if tc.accelerator.is_main_process:
#                     logging.info(f'Testing on testset for k={k_to_use}.')
#                 set_for_eval_with_topk(tc, k_to_use)
#                 test_loss, test_acc, total_average_flops, expert_average_costs, executed_expert_tokens, total_expert_tokens = online_evaluate_moe(
#                     tc.accelerator,
#                     tc.model,
#                     tc.test_loader,
#                     tc.criterion_type,
#                     cost_without_experts,
#                     token_expert_costs,
#                     batches=args.test_batches,
#                     return_counts=True)
#                 if tc.accelerator.is_main_process:
#                     final_losses.append(test_loss)
#                     final_scores.append(test_acc)
#                     # benchmark model efficiency
#                     final_flops.append(total_average_flops)
#                     final_expert_average_costs.append(expert_average_costs)
#                     final_expert_utilization.append(executed_expert_tokens)
#                     final_total_experts.append(total_expert_tokens)
#                     # log
#                     tc.writer.add_scalar(f'Eval with k={k_to_use}/Test loss', test_loss,
#                                          global_step=tc.state.current_batch)
#                     tc.writer.add_scalar(f'Eval with k={k_to_use}/Test accuracy', test_acc,
#                                          global_step=tc.state.current_batch)
#                     tc.writer.add_scalar(f'Eval with k={k_to_use}/Model FLOPs', total_average_flops,
#                                          global_step=tc.state.current_batch)
#         if tc.accelerator.is_main_process:
#             tc.writer.add_scalar('Eval/Model Params', model_params[''], global_step=tc.state.current_batch)
#             final_results = {}
#             final_results['args'] = args
#             final_results['model_state'] = unwrapped_model.state_dict()
#             final_results['test_losses'] = final_losses
#             final_results['hyperparam_values'] = args.dsti_tau_to_eval or args.k_to_eval
#             final_results['final_scores'] = final_scores
#             final_results['final_flops'] = final_flops
#             final_results['model_params'] = dict(model_params)
#             final_results['expert_average_costs'] = final_expert_average_costs
#             final_results['expert_utilization'] = final_expert_utilization
#             final_results['total_experts_used'] = final_total_experts
#             save_final(args, tc.final_path, final_results)


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
    setup_state(tc)
    make_vae(args, tc)

    tc.trainer = VARTrainer(
        device=args.device, # correct
        patch_nums=args.patch_nums, # correct
        resos=args.resos, # correct
        vae_local=tc.model_vae,
        var_wo_ddp=tc.model_var_wo_ddp, # correct
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
