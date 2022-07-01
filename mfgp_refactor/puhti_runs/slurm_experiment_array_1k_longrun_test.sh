#!/bin/bash
#SBATCH --job-name=activeLearning-D-1k
#SBATCH --account=project_2000382
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=longrun
#SBATCH --array=0-4
#SBATCH --output=activeLearning_1k_longrun_D_test_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dirs=(
"./AA_D_1k/run1"
"./AA_D_1k/run2"
"./AA_D_1k/run3"
"./AA_D_1k/run4"
"./AA_D_1k/run5"
)

module load python-data/3.7.6-1
export PROJAPPL=/projappl/project_2000382

if [ ! -d "${dirs[$SLURM_ARRAY_TASK_ID]}" ]; then
    # directory doesn't exist
    echo "run directories don't exist exitting !"
    exit 1
fi

cd ${dirs[$SLURM_ARRAY_TASK_ID]}
# srun ../main.sh $(cut -d / -f3 "${dirs[$SLURM_ARRAY_TASK_ID]}") input.dat
run_num=`cut -d / -f10 | pwd`
srun ../main_test.sh $run_num input.dat
