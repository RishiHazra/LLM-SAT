#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=72
#SBATCH --job-name="test-mixtral"
#SBATCH --output="slurm-%j.out"
#SBATCH --error="slurm-%j.err"
#SBATCH --mail-type="BEGIN,END,FAIL"
#SBATCH --mail-user="gabriele.venturato@kuleuven.be"

module load CUDA/11.7.1

# conda activate llmsat-env
conda activate llmast-new-env

export TRANSFORMERS_CACHE="$VSC_SCRATCH/.cache"
export HF_HOME="$VSC_SCRATCH/.cache"
python3 mixtral/inference.py
