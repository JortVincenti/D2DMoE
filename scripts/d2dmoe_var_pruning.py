import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.prune_var import train as pruned_var
from train import train
from utils import generate_run_name, submit_job


def load_env_variables(file_path):
    with open(file_path, 'r') as env_file:
        for line in env_file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

def main():
    # ════════════════════════ submitit setup ════════════════════════ #
    current_directory = os.getcwd()
    os.system('source /home/jvincenti/D2DMoE/user.env')
    load_env_variables('/home/jvincenti/D2DMoE/user.env')
    job_name = 'effbench'

    # account = 'plgccbench-gpu-a100'
    # account = 'plgimpmoe-gpu-a100'
    account = None

    # qos = 'normal'
    qos = None

    # partition = 'plgrid-gpu-a100'
    partition = 'gpu_h100'
    # partition = 'dgx'
    # partition = 'rtx3080'
    # partition = 'batch'

    timeout = 60 #60 * 24 * 7
    # timeout = 60 * 24 * 2

    gpus_per_task = 1
    gpu_type = ''
    # gpu_type = 'ampere:'
    cpus_per_gpu = None
    # cpus_per_gpu = 16
    mem_per_gpu = '16G'
    # mem_per_gpu = None

    executor = submitit.AutoExecutor(folder=os.environ['LOGS_DIR'])
    executor.update_parameters(
        stderr_to_stdout=True,
        timeout_min=timeout,
        slurm_job_name=job_name,
        slurm_account=account,
        slurm_qos=qos,
        slurm_partition=partition,
        slurm_ntasks_per_node=1,
        slurm_cpus_per_gpu=cpus_per_gpu,
        slurm_mem_per_gpu=mem_per_gpu,
    )

    # ════════════════════════ experiment settings ════════════════════════ #

    common_args = get_default_args()
    # exp_ids = [1, 2, 3]
    exp_ids = [1]
    common_args.runs_dir = Path(os.environ['RUNS_DIR'])
    common_args.dataset = 'TINYIMAGENET_PATH' #'imagenet'
    common_args.dataset_args = {}
    common_args.dataset_args.variant = 'deit3_rrc'# deit3 Jort added this would be for the train class to work for tiny deit3
    common_args.mixup_alpha = None #0.8
    common_args.cutmix_alpha = None #1.0
    common_args.mixup_smoothing = 0.1
    common_args.batch_size = 128
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'linear' #Jort: This should be 'linear'
    common_args.scheduler_args = {}
    common_args.epochs = 10
    common_args.scheduler_args.num_warmup_steps  = common_args.epochs * 1/50
    #common_args.scheduler_args.num_training_steps   = common_args.epochs * 1/50
    #common_args.scheduler_args.eta_min = 1e-6
    common_args.clip_grad_norm = 1.0
    
    common_args.eval_points = 20
    common_args.use_wandb = False
    common_args.mixed_precision = None

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # ════════════════════════ dsti router training model settings ════════════════════════ #

    # dsti_gpus_per_task = 4
    dsti_gpus_per_task = 1

    dsti_routing_args = deepcopy(common_args)
    dsti_routing_args.model_class = 'dsti_router'
    dsti_routing_args.router_loss_type = 'mse' #'huber' #'mse'
    dsti_routing_args.epochs = 1
    # dsti_routing_args.epochs = 0.1
    dsti_routing_args.batch_size = 256
    
    # dsti_routing_args.batch_size = 64
    dsti_routing_args.optimizer_args.lr = 0.001
    dsti_routing_args.model_args = {}
    dsti_routing_args.model_args.depth = 2
    dsti_routing_args.model_args.width = 128
    # dsti_routing_args.model_args.width = 32
    # dsti_routing_args.model_args.width = 16
    # dsti_routing_args.model_args.activation = 'gelu'
    dsti_routing_args.model_args.activation = 'gelu'
    # dsti_routing_args.model_args.activation = 'tanh'
    dsti_routing_args.model_args.output_activation = 'abs'
    # dsti_routing_args.model_args.output_activation = 'relu'
    # dsti_routing_args.model_args.output_activation = 'identity'
    dsti_routing_args.dsti_router_labels_layer = 'output'
    dsti_routing_args.dsti_router_labels_norm = 2
    
    #dsti_routing_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_routing_args.eval_points = 4
    # dsti_routing_args.eval_points = 0
    dsti_routing_args.mixed_precision = None
    #dsti_routing_args.mixed_precision = 'bf16'
    # # ════════════════════════ dsti router training ════════════════════════ #
    dsti_routing_args.batch_size_eff = 128
    dsti_routing_args.expert_index_switch = 8
    dsti_routing_args.pruning_percentage = [0.08, 0.06, 0.05, 0.03, 0.01,0.00]
    args = deepcopy(dsti_routing_args)
    executor.update_parameters(slurm_additional_parameters={})
    submit_job(executor, pruned_var, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

    # # ════════════════════════ plot cost vs acc ════════════════════════ #

    # plot_args = get_default_cost_plot_args()

    out_dir_name = f"vit_{common_args.dataset}_wip_hoyer"
    output_dir = Path(os.environ["RESULTS_DIR"]) / out_dir_name


if __name__ == '__main__':
    main()
