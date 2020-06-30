#!/bin/bash
#SBATCH --job-name=activeLearning-D-8k
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=batch
#SBATCH --array=1-5
#SBATCH --output=activeLearning-D-8k_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load anaconda/3
export PROJAPPL=/projappl/project_2000382

if [ ! -d "run$SLURM_ARRAY_TASK_ID" ]; then
    # directory doesn't exist
    echo "run directories don't exist exitting !"
    exit(1)
fi

cd run$SLURM_ARRAY_TASK_ID
srun ../main-D.sh run$SLURM_ARRAY_TASK_ID input.dat
