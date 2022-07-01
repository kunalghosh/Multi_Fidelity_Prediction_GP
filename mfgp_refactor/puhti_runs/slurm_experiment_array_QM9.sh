#!/bin/bash
#SBATCH --job-name=activeLearning-QM9-exp
#SBATCH --account=project_2000382
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=large
#SBATCH --array=0-4
#SBATCH --output=activeLearning_QM9_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dirs=(
"./QM9_D_EXP/run1"
"./QM9_D_EXP/run2"
"./QM9_D_EXP/run3"
"./QM9_D_EXP/run4"
"./QM9_D_EXP/run5"
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
srun ../main.sh $run_num input.dat
