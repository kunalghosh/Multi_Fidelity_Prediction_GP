#!/bin/bash
#SBATCH --time=00-04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24

srun gen_onehost.sh
