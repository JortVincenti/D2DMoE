#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=RunScript
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:10:00
#SBATCH --output=jobs/run.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $HOME/D2DMoE/

# # Empty the logs folder
# if [ -d "shared/results/effbench_logs" ]; then
#     rm -rf shared/results/effbench_logs/*
#     rm -rf shared/results/effbench_logs/.*
# fi

#!/bin/bash
source user.env
eval "$(conda shell.bash hook)"
conda activate effbench_env

#pip install --no-cache-dir plotly
# pip uninstall torch-fidelity
# pip install -e git+https://github.com/LTH14/torch-fidelity.git@master#egg=torch-fidelity

# pip install torch-fidelity
# pip install typed-argument-parser

#pip install wandb
# wandb login 94c83d220ddc780120eaa22226adf6730f644c6c

python -m scripts.$1