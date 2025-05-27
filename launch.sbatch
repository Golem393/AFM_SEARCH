#!/bin/bash
#SBATCH --job-name="myjob"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=PRACT
#SBATCH --qos=practical_course
#SBATCH --output=/storage/user/s0121/slurm/logs/slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go49jim@mytum.de

# Optional debug variables
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1

# Debug info (optional)
pwd; hostname; date
nvidia-smi

# Your Python command
python3 test.py