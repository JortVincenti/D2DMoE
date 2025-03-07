import logging
from typing import List

from omegaconf import OmegaConf
from torch import nn

from architectures.moe.dsti import dsti_mlp_filter_condition
from architectures.moe.moefication import replace_with_moes, split_original_parameters
from common import get_default_args, INIT_NAME_MAP
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state, make_vae
from utils import load_model
from utils_var import arg_util
from trainer import VARTrainer
from architectures.pretrained import get_var_d16 
from architectures.moe.dsti import find_gelu_activations, replace_with_relu
from pathlib import Path 
import torch
from utils import save_final


class ParamSplitContext(TrainingContext):
    orig_model: nn.Module = None
    replaced_modules_list: List[str] = None


def setup_model(args, tc):
    assert args.model_class == 'dsti_expert_split'

    if args.activation == 'gelu':
        base_model, tc.model_vae = get_var_d16() #load_model(args, args.base_on, args.exp_id)
        
        final_path = Path(args.path_file) #Path('/home/jvincenti/D2DMoE/shared/results/effbench_runs/TINYIMAGENET_PATH_enforce_sparsity_gelu_ft/final.pth')
        final_state = torch.load(final_path, map_location=args.device)
        state_dict = final_state['model_state'] 
        model_arg = final_state['args'].model_args
        activations_to_sparsify = find_gelu_activations(base_model, **model_arg)
        base_model = replace_with_relu(base_model, activations_to_sparsify)
        base_model = base_model.to(args.device)

        base_model.load_state_dict(state_dict)
        tc.orig_model = base_model
    elif args.activation == 'relu':
        base_model, tc.model_vae = get_var_d16() #load_model(args, args.base_on, args.exp_id)
        final_path =  Path(args.path_file) #Path('/home/jvincenti/D2DMoE/shared/results/effbench_runs/TINYIMAGENET_PATH_enforce_sparsity_relu_ft/final.pth')
        final_state = torch.load(final_path, map_location=args.device)
        state_dict = final_state['model_state'] 
        model_arg = final_state['args'].model_args
        base_model = base_model.to(args.device)
        base_model.load_state_dict(state_dict)
        tc.orig_model = base_model
    else:
        base_model, tc.model_vae = get_var_d16() #load_model(args, args.base_on, args.exp_id)
        tc.orig_model = base_model


    tc.model, tc.replaced_modules_list = replace_with_moes(base_model, **args.model_args,
                                                        module_filter_contition=dsti_mlp_filter_condition)
    print('done replace_with_moes', tc.replaced_modules_list)
    init_fun = INIT_NAME_MAP[args.init_fun]
    
    if init_fun is not None:
        init_fun(tc.model)
    tc.model = tc.accelerator.prepare(tc.model)


def param_split(tc):
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    print('tc.replaced_modules_list', tc.replaced_modules_list)
    if tc.state.current_batch == 0:
        split_original_parameters(tc.orig_model, unwrapped_model, tc.replaced_modules_list)
        tc.state.current_batch = max(tc.last_batch + 1, 1)
    del tc.orig_model


def final_eval(args, tc):
    if isinstance(args.runs_dir, str):
        args.runs_dir = Path("runs")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.final_path_save 
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tc.final_path = run_dir / 'final.pth'

    final_results = {}
    final_results['args'] = args
    final_results['model_state'] = tc.model.state_dict()

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
    tc = ParamSplitContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_state(tc)
    setup_data(args, tc)
    setup_optimization(args, tc)

    param_split(tc)

    # Build the trainer
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
    final_eval(args, tc)



def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
