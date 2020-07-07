#!/bin/bash
#SBATCH --job-name=activeLearning-A-4k
#SBATCH --account=project_2000382
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --tasks-per-node=20
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

module load anaconda/3
export PROJAPPL=/projappl/project_2000382

for SLURM_ARRAY_TASK_ID in ${!dirs[*]}
do
	if [ ! -d "${dirs[$SLURM_ARRAY_TASK_ID]}" ]; then
	    # directory doesn't exist
	    echo "run directories don't exist exitting !"
	    exit 1
	fi

	cd ${dirs[$SLURM_ARRAY_TASK_ID]}
	ls ../
	cd -
done
# srun ../main.sh run$SLURM_ARRAY_TASK_ID input.dat
