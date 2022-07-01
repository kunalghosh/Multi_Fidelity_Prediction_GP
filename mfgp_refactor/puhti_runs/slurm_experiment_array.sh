#!/bin/bash
#SBATCH --job-name=activeLearning-A-4k
#SBATCH --account=project_2000382
#SBATCH --time=60:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=large
#SBATCH --array=1-20
#SBATCH --output=activeLearning_%a.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

dirs=(
"./AA_A_4k/run1"
"./AA_A_4k/run2"
"./AA_A_4k/run3"
"./AA_A_4k/run4"
"./AA_A_4k/run5"
"./AA_A_8k/run1"
"./AA_A_8k/run2"
"./AA_A_8k/run3"
"./AA_A_8k/run4"
"./AA_A_8k/run5"
"./AA_D_4k/run1"
"./AA_D_4k/run2"
"./AA_D_4k/run3"
"./AA_D_4k/run4"
"./AA_D_4k/run5"
"./AA_D_8k/run1"
"./AA_D_8k/run2"
"./AA_D_8k/run3"
"./AA_D_8k/run4"
"./AA_D_8k/run5"
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
