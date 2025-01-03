22 Oct 2021

- Moved model files 13 to 16 to ghoshk1@taltta.aalto.fi:/m/phys/project3/cest/Kunal/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs/AA_A_1k/run1/
rsync --progress test-A-1k_13_model.pkl test-A-1k_14_model.pkl test-A-1k_15_full_idxs.npz test-A-1k_16_model.pkl ghoshk1@taltta.aalto.fi:/m/phys/project3/cest/Kunal/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs/AA_A_1k/run1/

- Deleted the model files

- Created a folder "old_index_files" where all index files from batch 12 onwards were moved. This was done so that I can compare the new indices to 11 and 12 (which were run in a single run from the beginning).
mv test-A-1k_12_full_idxs.npz test-A-1k_13_full_idxs.npz test-A-1k_13_idxs.npz test-A-1k_14_full_idxs.npz test-A-1k_14_idxs.npz test-A-1k_15_full_idxs.npz test-A-1k_15_idxs.npz test-A-1k_16_full_idxs.npz test-A-1k_16_idxs.npz test-A-1k_17_idxs.npz old_index_files

2 Nov 2021

- Moved model file 11, 12 and 13 to taltta, kept 14 to help resume
rsync --progress test-A-1k_11_model.pkl test-A-1k_12_model.pkl test-A-1k_13_model.pkl  ghoshk1@taltta.aalto.fi:/m/phys/project3/cest/Kunal/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs/AA_A_1k/run1/

