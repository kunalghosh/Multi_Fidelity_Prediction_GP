#!/bin/bash
#SBATCH --time=00-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=800

srun gen_onehost.sh
