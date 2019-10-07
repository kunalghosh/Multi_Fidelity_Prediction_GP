#!/bin/bash -e
cd $WRKDIR/MBTR_62k_GEN/Multi_Fidelity_Prediction_GP/mfgp/task1
mkdir build || true
cd build
module load CUDA/9.0.176.lua || true
module load anaconda3 || true
source activate ../../task0/build_mbtr/myenv
# python ../GP_prediction_low_fidelity_ExactGPyTorch.py ../../task0/mbtr_data_flattened/mbtr.npz ../../task0/build_mbtr/df_full_split_ids_with_smiles_v15.json
python ../GP_prediction_low_fidelity_ExactGPyTorch.py ../../task0/build_mbtr/mbtrk2.npz ../../task0/build_mbtr/df_full_split_ids_with_smiles_v15.json
