#!/bin/bash
#SBATCH --job-name=activeLearning-A-1k_mae_1
#SBATCH --account=project_2000382
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=small
#SBATCH --array=0
#SBATCH --output=activeLearning_1k_longrun_test_mae_run1_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dirs=(
"12"
"13"
)

module load python-data/3.7.6-1
export PROJAPPL=/projappl/project_2000382

# if [ ! -d "${dirs[$SLURM_ARRAY_TASK_ID]}" ]; then
#     # directory doesn't exist
#     echo "run directories don't exist exitting !"
#     exit 1
# fi

# cd ${dirs[$SLURM_ARRAY_TASK_ID]}
# srun ../main.sh $(cut -d / -f3 "${dirs[$SLURM_ARRAY_TASK_ID]}") input.dat
# run_num=`cut -d / -f10 | pwd`
srun run_getmae.sh ${dirs[$SLURM_ARRAY_TASK_ID] }
