import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial
from dataclasses import dataclass
import logging
import math
from copy import deepcopy
from datetime import datetime
from pathlib import PurePath, Path
from typing import Any, Callable, List, Type, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from common import INIT_NAME_MAP, LOSS_NAME_MAP, OPTIMIZER_NAME_MAP, SCHEDULER_NAME_MAP, get_default_args
from data_utils.data import DATASETS_NAME_MAP
from eval import benchmark, test_classification
from utils import (
    Mixup,
    create_model,
    generate_run_name,
    get_loader,
    get_lrs,
    get_run_id,
    load_state,
    save_final,
    save_state,
    unfrozen_parameters,
    load_model,
)

import dist
from utils_var import arg_util, misc
from utils_var.data import build_dataset
from utils_var.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils_var.misc import auto_resume
from trainer import VARTrainer
from utils_var.amp_sc import AmpOptimizer
from utils_var.lr_control import filter_params, lr_wd_annealing
from architectures import VAR, VQVAE, build_vae_var




def main_training(args=None, tc=None):   
    args = args or arg_util.init_dist_and_get_args()
    tc = tc or TrainingContext()

    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
        
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    trainer = tc.trainer
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    iters_train = 10
    ld_train, ld_val = tc.train_loader, tc.val_loader

    tc.trainer_state = trainer_state
    # train
    start_time = time.time()
    best_L_mean, best_L_tail, best_acc_mean, best_acc_tail = 999., 999., -1., -1.
    best_val_loss_mean, best_val_loss_tail, best_val_acc_mean, best_val_acc_tail = 999, 999, -1, -1
    
    L_mean, L_tail = -1, -1
    print("Epochs", start_ep, start_it, args.ep)
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)
        
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )
        
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        best_L_mean, best_acc_mean = min(best_L_mean, L_mean), max(best_acc_mean, acc_mean)
        if L_tail != -1: best_L_tail, best_acc_tail = min(best_L_tail, L_tail), max(best_acc_tail, acc_tail)
        args.L_mean, args.L_tail, args.acc_mean, args.acc_tail, args.grad_norm = L_mean, L_tail, acc_mean, acc_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        
        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail)
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, val_acc_mean), max(best_val_acc_tail, val_acc_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail, vacc_mean=val_acc_mean, vacc_tail=val_acc_tail)
            args.vL_mean, args.vL_tail, args.vacc_mean, args.vacc_tail = val_loss_mean, val_loss_tail, val_acc_mean, val_acc_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s')
            
            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()
        
        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Acc m&t: {best_acc_mean:.2f} {best_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer):
    # import heavy packages after Dataloader object creation    
    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train
    
    for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()
        
        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)
        
        args.cur_it = f'{it+1}/{iters_train}'
        
        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd
        
        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1
        
        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)
        
        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )
        
        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost

@dataclass
class TrainingState:
    current_batch: int = 0

    def state_dict(self):
        return {'current_batch': self.current_batch}

    def load_state_dict(self, state_dict):
        self.current_batch = state_dict['current_batch']

# Original train.py functions remain unchanged...
@dataclass
class TrainingContext:
    accelerator: Accelerator = None
    model: torch.nn.Module = None
    model_vae: torch.nn.Module = None
    model_var_wo_ddp: torch.nn.Module = None
    trainer = None
    # savable state
    state: TrainingState = None
    trainer_state = None
    # data
    train_loader: torch.utils.data.DataLoader = None
    train_eval_loader: torch.utils.data.DataLoader = None
    test_loader: torch.utils.data.DataLoader = None
    last_batch: int = None
    eval_batch_list: List[int] = None
    # optimization
    criterion_type: Type = None
    criterion: Callable = None
    optimizer: torch.optim.Optimizer = None
    scheduler: Any = None
    teacher_model: torch.nn.Module = None
    distill_weight: float = None
    # files and logging
    state_path: PurePath = None
    final_path: PurePath = None
    writer: SummaryWriter = None

def setup_accelerator(args, tc): # Jort: This should be correct!
    """Set up the accelerator for training using distributed and mixed-precision settings."""
    # Jort: Comment, not sure if this is needed for the training loop.
    # do not change the split_batches argument
    # resuming a run with different resources and split_batches=False would cause the batches_per_epoch to be different
    tc.accelerator = Accelerator(split_batches=True,
                                 # overwrite the unintuitive behaviour of accelerate, see:
                                 # https://discuss.huggingface.co/t/learning-rate-scheduler-distributed-training/30453
                                 step_scheduler_with_optimizer=False,
                                 mixed_precision=args.mixed_precision, cpu=args.cpu,
                                 # find_unused_parameters finds unused parameters when using DDP
                                 # but may affect performance negatively
                                 # sometimes it happens that somehow it works with this argument, but fails without it
                                 # so - do not remove
                                 # see: https://github.com/pytorch/pytorch/issues/43259
                                 kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)] # Jort: Default is (find_unused_parameters=True)
                                 )


def compile_model(m, fast):
    if fast == 0:
        return m
    return torch.compile(m, mode={
        1: 'reduce-overhead',
        2: 'max-autotune',
        3: 'default',
    }[fast]) if hasattr(torch, 'compile') else m

def setup_model(args, tc): # Jort: This should be correct!
    model = create_model(args.model_class, args.model_args)
    init_fun = INIT_NAME_MAP[args.init_fun]
    if init_fun is not None:
        init_fun(model)
    tc.model = tc.accelerator.prepare(model)
    tc.model.train()

def setup_data(args, tc): # Jort: This should be correct!

    batch_size = args.batch_size
    if args.test_batch_size is None:
        test_batch_size = batch_size
    else:
        test_batch_size = args.test_batch_size
    
    
    """Prepare the data loaders for training and validation."""
    num_classes, train_dataset, val_dataset = build_dataset(args.data_path, args.data_load_reso, args.hflip, args.mid_reso)
    types = str((type(train_dataset).__name__, type(val_dataset).__name__))

    tc.val_loader = tc.accelerator.prepare(DataLoader(
        val_dataset, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    ))
    tc.test_loader = tc.val_loader # Jort: For now keep it like this.
    del val_dataset

    _, start_ep, start_it, _, _ = auto_resume(args, 'ar-ckpt*.pth')
    
    tc.train_loader  =  tc.accelerator.prepare(DataLoader(
        dataset=train_dataset, num_workers=args.workers, pin_memory=True,
        generator=None, # Jort, For now keep it none original: get_different_generator_for_each_rank(args), # worker_init_fn=worker_init_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(train_dataset), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
        ),
    ))
    del train_dataset

    if args.last_batch is None:
        batches_per_epoch = len(tc.train_loader)
        tc.last_batch = math.ceil(args.epochs * batches_per_epoch - 1)
        if tc.accelerator.is_main_process:
            logging.info(f'{args.epochs=} {batches_per_epoch=} {tc.last_batch=}')
    else:
        tc.last_batch = args.last_batch
        if tc.accelerator.is_main_process:
            logging.info(f'{tc.last_batch=}')
    tc.eval_batch_list = [
        round(x) for x in torch.linspace(0, tc.last_batch, steps=args.eval_points, device='cuda').tolist()
    ]
 
def setup_optimization(args, tc): # Jort: This should be roughly correct (tc.optimizer, tc.scheduler)!
    """Set up the optimizer and learning rate scheduler."""


    # Create a full list of parameter names from the model
    # all_params = set([name for name, _ in tc.model.named_parameters()])

    # # Compute the parameters to freeze by subtracting the ones to unfreeze
    # params_to_freeze = all_params - params_to_unfreeze
    # names, paras, para_groups = filter_params(tc.model, nowd_keys=params_to_freeze)

    names, paras, para_groups = filter_params(tc.model, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)

    criterion_args = args.loss_args
    tc.criterion_type = LOSS_NAME_MAP[args.loss_type]
    tc.criterion = tc.criterion_type(reduction='mean', **criterion_args)
    optimizer_args = args.optimizer_args

    if 'selective_weight_decay' in optimizer_args and optimizer_args['selective_weight_decay']:
        param_dict = {pn: p for pn, p in tc.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        params = [
            {'params': decay_params, 'weight_decay': optimizer_args.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        del optimizer_args['weight_decay']
        del optimizer_args['selective_weight_decay']
    else:
        params = unfrozen_parameters(tc.model)

    tc.optimizer = tc.accelerator.prepare(AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )) # Jort: Originaly this should be wrapped into tc.accelerator.prepare() but this is not possible with AmpOptimizer
    del names, paras, para_groups

    if args.scheduler_class is not None:
        scheduler_args = deepcopy(args.scheduler_args)
        if 'patience' in scheduler_args:
            scheduler_args['patience'] = int(scheduler_args['patience'] * tc.last_batch)
        if args.scheduler_class == 'cosine':
            scheduler_args['T_max'] = tc.last_batch
        if args.scheduler_class in ['cosine_with_warmup', 'linear', 'inverse_sqrt']:
            scheduler_args['num_training_steps'] = tc.last_batch
        tc.scheduler = tc.accelerator.prepare(SCHEDULER_NAME_MAP[args.scheduler_class](tc.optimizer, **scheduler_args)) #Jort this should be wrapped into tc.accelerator.prepare() but this is not possible with SCHEDULER_NAME_MAP

    if args.distill_from is not None:
        teacher_model, _, _ = load_model(args, args.distill_from, args.exp_id)
        tc.teacher_model = tc.accelerator.prepare(teacher_model)
        tc.teacher_model.eval()
        tc.distill_weight = 1.0 if args.distill_weight is None else args.distill_weight
        # TODO optionally parameterize the type of loss
        tc.distill_criterion = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')


def training_loop(args, tc):
    """Execute the main training loop."""
    main_training(args, tc)

def validate(tc):
    raise NotImplementedError("Validation is not implemented yet.")
    """Run validation on the validation set."""
    L_mean, L_tail, acc_mean, acc_tail, tot, duration = tc.trainer.eval_ep(tc.val_loader)

def setup_files_and_logging(args, tc): # Jort: This should be correct!
    # files setup
    if isinstance(args.runs_dir, str):
        args.runs_dir = Path("runs")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    _, run_name = generate_run_name(args)
    run_dir = args.runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tc.state_path = run_dir / 'state'
    tc.final_path = run_dir / 'final.pth'
    # logging setup
    if tc.accelerator.is_main_process:
        # log config
        logging.info(f'{run_name} args:\n{args}')
        if args.use_wandb:
            entity = os.environ['WANDB_ENTITY']
            project = os.environ['WANDB_PROJECT']
            run_id = get_run_id(run_name)
            wandb.tensorboard.patch(root_logdir=str(run_dir.resolve()), pytorch=True, save=False)
            if run_id is not None:
                wandb.init(entity=entity, project=project, id=run_id, resume='must', dir=str(run_dir.resolve()))
            else:
                wandb.init(entity=entity, project=project, config=dict(args), name=run_name,
                           dir=str(run_dir.resolve()))
            wandb.run.log_code('.', include_fn=lambda path: path.endswith('.py'))
        tc.writer = SummaryWriter(str(run_dir.resolve()))


def in_training_eval(args, tc):

    """Perform evaluation during training at specific intervals."""
    if tc.state.current_batch in tc.eval_batch_list:
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Train/Progress',
                                 tc.state.current_batch / tc.last_batch,
                                 global_step=tc.state.current_batch)
        test_loss, test_acc = test_classification(tc.accelerator,
                                                  tc.model,
                                                  tc.test_loader,
                                                  tc.criterion_type,
                                                  tc, 
                                                  batches=args.eval_batches)
        if tc.accelerator.is_main_process and hasattr(tc, 'sparsity'):
            tc.writer.add_scalar('Eval/Test sparsity', tc.sparsity, global_step=tc.state.current_batch)
        # train_loss, train_acc = test_classification(tc.accelerator,
        #                                             tc.model,
        #                                             tc.train_eval_loader,
        #                                             tc.criterion_type,
        #                                             tc,
        #                                             batches=args.eval_batches)
        # if tc.accelerator.is_main_process and hasattr(tc, 'sparsity'):
        #     tc.writer.add_scalar('Eval/Train sparsity', tc.sparsity, global_step=tc.state.current_batch)
        if tc.accelerator.is_main_process:
            tc.writer.add_scalar('Eval/Test loss', test_loss, global_step=tc.state.current_batch)
            tc.writer.add_scalar('Eval/Test accuracy', test_acc, global_step=tc.state.current_batch)
            # tc.writer.add_scalar('Eval/Train loss', train_loss, global_step=tc.state.current_batch)
            # tc.writer.add_scalar('Eval/Train accuracy', train_acc, global_step=tc.state.current_batch)

def setup_state(tc):
    tc.state = TrainingState()
    tc.accelerator.register_for_checkpointing(tc.state)
    load_state(tc.accelerator, tc.state_path)

def final_eval(args, tc):
    """Use the eval_ep method from VARTrainer for final evaluation."""
    L_mean, L_tail, acc_mean, acc_tail, tot, duration, model_costs, model_params = tc.trainer.eval_ep(tc.val_loader)
    print(f"Final evaluation completed: \n"
          f"Mean Loss: {L_mean:.4f}, Tail Loss: {L_tail:.4f}\n"
          f"Mean Accuracy: {acc_mean:.2f}%, Tail Accuracy: {acc_tail:.2f}%\n"
          f"Total samples: {tot}, Duration: {duration:.2f}s")

    final_results = {}
    final_results['args'] = args
    unwrapped_model = tc.accelerator.unwrap_model(tc.model)
    final_results['model_state'] = unwrapped_model.state_dict()
    tc.writer.add_scalar('Eval/Test loss', L_mean)
    tc.writer.add_scalar('Eval/Test accuracy', acc_mean)
    final_results['final_score'] = acc_mean
    final_results['final_loss'] = L_mean
    logging.info(f'Test accuracy: {acc_mean}')

    final_results['model_flops'] = model_costs.total()
    final_results['model_flops_by_module'] = dict(model_costs.by_module())
    final_results['model_flops_by_operator'] = dict(model_costs.by_operator())
    final_results['model_params'] = dict(model_params)
    tc.writer.add_scalar('Eval/Model FLOPs', model_costs.total())
    tc.writer.add_scalar('Eval/Model Params', model_params[''])
    save_final(args, tc.final_path, final_results)

def make_vae(args, tc):
    
    # build models 
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  
    vae_local, _ = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=patch_nums,
        num_classes=1000, depth=16, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, only_vae=True
    )
    
    vae_ckpt = 'vae_ch160v4096z32.pth'
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cuda'), strict=True)
    
    vae_local: VQVAE = compile_model(vae_local, args.vfast)
    tc.model_vae = vae_local

    

def train(args):
    args: arg_util.Args = arg_util.init_dist_and_get_args(args)
    tc = TrainingContext()
    setup_accelerator(args, tc)
    setup_files_and_logging(args, tc)
    setup_model(args, tc)
    setup_data(args, tc)
    setup_optimization(args, tc)
    setup_state(tc)
    make_vae(args, tc)
    
    if args.epochs == 0:
        var_ckpt = f'var_d16.pth'
        tc.model_var_wo_ddp.load_state_dict(torch.load(var_ckpt, map_location='cuda'), strict=True)
        var: DDP = (DDP if dist.initialized() else NullDDP)(tc.model_var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
        tc.model = var
        # Build the trainer
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
        final_eval(args, tc)
    else:
        # Build the trainer
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
        if tc.trainer_state is not None and len(tc.trainer_state):
            tc.trainer.load_state_dict(tc.trainer_state, strict=False, skip_vae=True) # don't load vae again
       
        training_loop(args, tc)
        final_eval(args, tc)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == '__main__':
    try: main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
