#!/bin/bash
#SBATCH --time=00-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
# --mem-per-cpu=2048
#SBATCH --gres=gpu:v100:1
# --mem-per-cpu=409600

srun run_gp.sh
