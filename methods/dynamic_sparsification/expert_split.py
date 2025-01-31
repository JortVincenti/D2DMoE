import logging
from typing import List

from omegaconf import OmegaConf
from torch import nn

from architectures.moe.dsti import dsti_mlp_filter_condition
from architectures.moe.moefication import replace_with_moes, split_original_parameters
from common import get_default_args, INIT_NAME_MAP
from train import TrainingContext, setup_accelerator, setup_data, setup_optimization, setup_files_and_logging, \
    setup_state, final_eval, make_vae
from utils import load_model
from utils_var import arg_util
from trainer import VARTrainer    


class ParamSplitContext(TrainingContext):
    orig_model: nn.Module = None
    replaced_modules_list: List[str] = None


def setup_model(args, tc):
    assert args.model_class == 'dsti_expert_split'
    base_model, base_args, _, tc.var_wo_ddp = load_model(args, args.base_on, args.exp_id)
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
    print('unwrapped_model', unwrapped_model)
    print('tc.orig_model', tc.orig_model)
    print('tc.replaced_modules_list', tc.replaced_modules_list)
    if tc.state.current_batch == 0:
        split_original_parameters(tc.orig_model, unwrapped_model, tc.replaced_modules_list)
        tc.state.current_batch = max(tc.last_batch + 1, 1)
    del tc.orig_model


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
    print('5')
    setup_data(args, tc)
    print('6')
    setup_optimization(args, tc)
    print('7')
    setup_state(tc)

    make_vae(args, tc)
    # Build the trainer
    tc.trainer = VARTrainer(
        device=args.device, # correct
        patch_nums=args.patch_nums, # correct
        resos=args.resos, # correct
        vae_local=tc.model_vae,
        var_wo_ddp=tc.var_wo_ddp, # correct
        var=tc.model, # correct
        var_opt=tc.optimizer, # correct
        label_smooth=args.ls # correct
    )

    print('8')
    param_split(tc)

    tc.model=tc.model.to('cuda')

    print('9')
    final_eval(args, tc)



def main():
    args = OmegaConf.merge(get_default_args(), OmegaConf.from_cli())
    train(args)


if __name__ == '__main__':
    main()
