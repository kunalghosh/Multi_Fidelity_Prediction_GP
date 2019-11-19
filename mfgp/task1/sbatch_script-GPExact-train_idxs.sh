#!/bin/bash
#SBATCH --time=00-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4096
# --mem-per-cpu=409600

srun run_sklearn-train_idxs.sh
