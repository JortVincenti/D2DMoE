import os
from copy import deepcopy
from pathlib import Path

import submitit

from common import get_default_args
from methods.avit import train as train_avit
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
from methods.dynamic_sparsification.find_best_tau import train as best_tau
from methods.dynamic_sparsification.find_best_tau_image import train as best_tau_image
from methods.early_exit import train as train_ee
from methods.moefication.expert_split import train as split_moe
from methods.moefication.relufication import train as relufication
from methods.moefication.train_routers import train as train_routers
from train import train
from utils import generate_run_name, submit_job
#import utils
from visualize.activation_sparsity import get_default_args as get_default_activation_sparsity_args
from visualize.activation_sparsity import main as activation_sparsity_plot
from visualize.cost_vs_plot import get_default_args as get_default_cost_plot_args
from visualize.cost_vs_plot import main as cost_vs_plot
from visualize.dsti_one_at_a_time import get_default_args as get_default_dsti_one_at_a_time_args
from visualize.dsti_one_at_a_time import main as dsti_one_at_a_time_plot
from visualize.moe_image_spatial_load import get_default_args as get_default_moe_image_spatial_load_args
from visualize.moe_image_spatial_load import main as dsti_spatial_load_plot


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
    partition = 'gpu_a100'
    # partition = 'dgx'
    # partition = 'rtx3080'
    # partition = 'batch'

    timeout = 12*60 #60 * 24 * 7
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
    dsti_routing_args.epochs = 3
    # dsti_routing_args.epochs = 0.1
    # dsti_routing_args.batch_size = 256
    dsti_routing_args.batch_size = 128
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
    dsti_routing_args.dsti_tau_to_eval = [0.9, 0.925, 0.93, 0.935, 0.94, 0.945,0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]

    dsti_routing_args.dsti_expert_selection_mode = 'dynk_max'
    dsti_routing_args.eval_points = 4
    # dsti_routing_args.eval_points = 0
    dsti_routing_args.mixed_precision = None
    dsti_routing_args.fid = False
    dsti_routing_args.debug = False
    dsti_routing_args.use_router = False
    #dsti_routing_args.mixed_precision = 'bf16'
    # Include Sparsity or not

    final_path_save = [
        'relu_data_0',
       #'relu_data_0.1',
        #'relu_data_0.01',
        #'relu_data_0.001',
        #'relu_data_0.0001',
    ]

    path_file_ft = [
        '/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_sparse_ft_0/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_sparse_ft_0.1/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_sparse_ft_0.01/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_sparse_ft_0.001/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_sparse_ft_0.0001/final.pth',

    ]

    path_file_moe = [
        '/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_moe_0/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_moe_0.1/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_moe_0.01/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_moe_0.001/final.pth',
        #'/home/jvincenti/D2DMoE/shared/results/effbench_runs/relu_moe_0.0001/final.pth',
    ]


    # ════════════════════════ dsti router training ════════════════════════ #
    base_routed_dsti_exp_names = []

    for exp_id in exp_ids:
        for i in range(len(final_path_save)):

            dsti_routing_args.path_file_moe = path_file_moe[i]
            dsti_routing_args.final_path_save = final_path_save[i]

            if dsti_routing_args.final_path_save == 'base_data_moe':
                dsti_routing_args.activation = None
            elif 'relu' in dsti_routing_args.final_path_save:
                dsti_routing_args.activation = 'relu'
                dsti_routing_args.path_file_ft = path_file_ft[i]
            else:
                dsti_routing_args.activation = 'gelu'
                dsti_routing_args.path_file_ft = path_file_ft[i]

            args = deepcopy(dsti_routing_args)
            args.exp_id = exp_id
            args.base_on = 'base_exp'
            exp_name, run_name = generate_run_name(args)
            base_run_name = f'{args.base_on}_{exp_id}'
            executor.update_parameters(slurm_additional_parameters={})
            job = submit_job(executor, best_tau_image, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            job_2 = submit_job(executor, best_tau, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
            jobs.append(job)
            jobs.append(job_2)
            run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_routed_dsti_exp_names.append(exp_names[-1])
        display_names.append(f'DSTI')

    # ═════════════════════════════════════════════════════════ #

    print(f"Exp names: {exp_names}")
    print(f"Display names: {display_names}")
    print(f"SLURM JIDs: {[job.job_id for job in jobs]}")

if __name__ == '__main__':
    main()
