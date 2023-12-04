#!/bin/bash -l
#SBATCH --clusters=wice
#SBATCH --partition=gpu
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=72
#SBATCH --job-name="llm-sat"
#SBATCH --output="slurm-%j.out"
#SBATCH --error="slurm-%j.err"
#SBATCH --mail-type="BEGIN,END,FAIL"

module load CUDA/11.7.1

conda activate llmsat-env

export TRANSFORMERS_CACHE="$VSC_SCRATCH/.cache"
python3 llama/inference.py
