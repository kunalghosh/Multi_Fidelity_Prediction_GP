#!/bin/bash
#SBATCH --job-name=activeLearning-D
#SBATCH --account=project_2000382
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=large
#SBATCH --output=activeLearning-D.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load bioconda/3
export PROJAPPL=/projappl/project_2000382

srun main-D.sh
