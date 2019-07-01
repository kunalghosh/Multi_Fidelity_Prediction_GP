#!/bin/bash
#SBATCH --time=00-04
#SBATCH --nodes=1
#SBATCH --ntasks=24

git clone https://github.com/kunalghosh/Multi_Fidelity_Prediction_GP.git
cd Multi_Fidelity_Prediction_GP/task0
srun gen_onehost.sh
