import os
from copy import deepcopy
from pathlib import Path
import submitit
from common import get_default_args
from methods.dynamic_sparsification.expert_split import train as dsti_expert_split
from methods.dynamic_sparsification.rep_distill_train import train as mha_distill
from methods.dynamic_sparsification.sparse_finetuning import train as sparse_finetune
from methods.dynamic_sparsification.train_routers import train as dsti_train_routers
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

    timeout = 60 * 20
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
    common_args.batch_size = 16 #128
    common_args.loss_type = 'ce'
    common_args.loss_args = {}
    common_args.optimizer_class = 'adam'
    common_args.optimizer_args = {}
    common_args.optimizer_args.lr = 0.001
    common_args.optimizer_args.weight_decay = 0.0
    common_args.scheduler_class = 'cosine' #Jort: This should be 'linear'
    common_args.scheduler_args = {}
    common_args.scheduler_args.eta_min = 1e-6
    common_args.clip_grad_norm = 1.0
    common_args.epochs = 5
    common_args.eval_points = 20
    common_args.use_wandb = False
    common_args.mixed_precision = None

    jobs = []
    run_to_job_map = {}
    exp_names = []
    display_names = []

    # # ════════════════════════ base model settings ════════════════════════ #

    base_model_args = deepcopy(common_args)
    base_model_args.model_class = 'var_d16' #'tv_vit_b_16'
    base_model_args.model_args = {}
    base_model_args.epochs = 0  # pretrained
    base_model_args.eval_points = 0
    
    # # ════════════════════════ get pretrained base models ════════════════════════ #
    # Jort: First thing needs to run 
    for exp_id in exp_ids:
        args = deepcopy(base_model_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        # job = submit_job(executor, train, args, num_gpus=gpus_per_task, gpu_type=gpu_type)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    display_names.append(f'VAR')
    base_exp_name = exp_name
    # # ════════════════════════ MHA replacement model settings ════════════════════════ #
    # # Jort: Second job to run 
    dsti_gpus_per_task = 1 #2
    # dsti_gpus_per_task = 3
    # dsti_gpus_per_task = 4

    distillation_args = deepcopy(common_args)
    distillation_args.model_class = 'mha_rep_distill'
    distillation_args.model_args = {}
    distillation_args.model_args.flops_factor = 1.0
    distillation_args.model_args.mlp_type = 'simple'
    # distillation_args.model_args.mlp_type = 'residual'
    distillation_args.model_args.dropout = 0.05
    # distillation_args.model_args.dropout = 0.0
    distillation_args.base_on = base_exp_name
    distillation_args.epochs = 3
    # distillation_args.epochs = 1
    # distillation_args.epochs = 0
    # distillation_args.epochs = 0.1
    distillation_args.batch_size = 64
    # distillation_args.batch_size = 64
    distillation_args.dsti_distill_loss_type = 'mse'
    distillation_args.eval_points = 0
    distillation_args.optimizer_args.lr = 0.001
    #
    # distillation_args.dsti_enforce_weight = 1e-2
    distillation_args.dsti_enforce_weight = 1e-3
    # distillation_args.dsti_enforce_weight = 8e-4
    # distillation_args.dsti_enforce_weight = 1e-4
    # distillation_args.dsti_enforce_weight = 1e-5
    distillation_args.dsti_enforce_mode = 'relu_hoyer'
    # distillation_args.dsti_enforce_mode = 'relu_l1'
    distillation_args.dsti_enforce_schedule = 'linear'
    # doesn't work with mixed precision yet
    # distillation_args.mixed_precision = 'bf16'

    # # # ════════════════════════ MHA distillation into non-linear QKVO modules ════════════════════════ #
    base_distill_exp_names = []
    for exp_id in exp_ids:
        args = deepcopy(distillation_args)
        args.exp_id = exp_id
        exp_name, run_name = generate_run_name(args)
        base_run_name = f'{base_exp_name}_{exp_id}'
        # if base_run_name in run_to_job_map:
        #     dependency_str = f"afterany:{run_to_job_map[base_run_name].job_id}"
        #     executor.update_parameters(slurm_additional_parameters={"dependency": dependency_str})
        # else:
        #     executor.update_parameters(slurm_additional_parameters={})
        # job = submit_job(executor, mha_distill, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
        # jobs.append(job)
        # run_to_job_map[run_name] = job
    exp_names.append(exp_name)
    base_distill_exp_names.append(exp_names[-1])
    display_names.append(f'MHA distillation')

    # # ════════════════════════ sparsity enforcement settings ════════════════════════ #
    # Jort: Third Job to run
    # dsti_gpus_per_task = 4
    # dsti_gpus_per_task = 3
    dsti_gpus_per_task = 1

    sparsity_enforcement_args = deepcopy(common_args)
    sparsity_enforcement_args.dsti_enforce_mode = 'relu_hoyer'
    sparsity_enforcement_args.dsti_clamp_displacement = -10.0
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-1
    dsti_enforce_weight = [1e-4] #, 1e-3, 1e-4] #[0, 2e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 2e-4] 
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-1
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-2
    # sparsity_enforcement_args.dsti_enforce_weight = 5e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-3
    # sparsity_enforcement_args.dsti_enforce_weight = 2e-4
    # sparsity_enforcement_args.dsti_enforce_weight = 1e-4
    #sparsity_enforcement_args.dsti_enforce_schedule = 'linear'
    sparsity_enforcement_args.dsti_enforce_schedule = None #'linear'
    sparsity_enforcement_args.model_class = 'enforce_sparsity'
    sparsity_enforcement_args.model_args = {}
    sparsity_enforcement_args.model_args.apply_to = 'moe_eligible_only'
    sparsity_enforcement_args.epochs = 2
    # sparsity_enforcement_args.epochs = 0.1
    #sparsity_enforcement_args.batch_size = 256
    sparsity_enforcement_args.batch_size = 512
    sparsity_enforcement_args.eval_points = 46
    # sparsity_enforcement_args.eval_points = 0
    sparsity_enforcement_args.optimizer_args.lr = 2e-5
    # sparsity_enforcement_args.optimizer_args.lr = 2e-4
    # sparsity_enforcement_args.optimizer_args.lr = 5e-4
    sparsity_enforcement_args.optimizer_args.weight_decay = 0.05
    # end LR for scheduler
    sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 1e-5
    # sparsity_enforcement_args.scheduler_args.eta_min = 2e-6
    #
    sparsity_enforcement_args.final_path_save = 'relu_sparse_ft_d24'
    sparsity_enforcement_args.mixed_precision = 'bf16'
    #sparsity_enforcement_args.mixed_precision = 'bf16'
    sparsity_enforcement_args.path_file_ft = 'shared/results/effbench_runs/relu_sparse_ft_0/final.pth'

    # # ════════════════════════ activation sparsity enforcement ════════════════════════ #
    # Jort HEre
    base_enforce_exp_names = []
    for base_on_exp_name in base_distill_exp_names:
        # for base_on_exp_name in [base_exp_name]:
        for exp_id in exp_ids:
            for weight in dsti_enforce_weight:
                sparsity_enforcement_args.dsti_enforce_weight = weight
                args = deepcopy(sparsity_enforcement_args)
                args.exp_id = exp_id
                args.base_on = base_on_exp_name
                exp_name, run_name = generate_run_name(args)
                base_run_name = f'{base_on_exp_name}_{exp_id}'
                executor.update_parameters(slurm_additional_parameters={})
                job = submit_job(executor, sparse_finetune, args, num_gpus=dsti_gpus_per_task, gpu_type=gpu_type)
                jobs.append(job)
                run_to_job_map[run_name] = job
        exp_names.append(exp_name)
        base_enforce_exp_names.append(exp_names[-1])
        display_names.append(f'Sparsity enforcement')

if __name__ == '__main__':
    main()
