#!/bin/bash
#SBATCH --time=00-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=409600

srun run_gp.sh
