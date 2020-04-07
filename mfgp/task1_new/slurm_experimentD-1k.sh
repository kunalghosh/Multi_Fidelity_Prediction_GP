#!/bin/bash
#SBATCH --job-name=activeLearning-D-1k
#SBATCH --account=project_2000382
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#  SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --partition=small
#SBATCH --output=activeLearning-D-1k-temp.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load bioconda/3
export PROJAPPL=/projappl/project_2000382

srun main-D-1k.sh
