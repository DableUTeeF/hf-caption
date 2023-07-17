#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus=1		                # Specify total number of GPUs
#SBATCH -t 120:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt200060                     # Specify project name
#SBATCH -J cap_train                      # Specify job name

#SBATCH --error=slurm_outputs/%j.out
#SBATCH --output=slurm_outputs/%j.out

module purge
module load Miniconda3/22.11.1-1
conda activate /project/lt200060-capgen/palm/conda_envs/.conda/envs/palm_caption

export _TYPER_STANDARD_TRACEBACK=1
export TYPER_STANDARD_TRACEBACK=1
export ACCELERATE_DISABLE_RICH=1

srun python3 train_baseline.py
