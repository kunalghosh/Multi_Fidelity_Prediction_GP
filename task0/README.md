# Data generation
We have access to the xyz files (In the Pandas dataframe, stored as a json). 
The scripts in this directory convert the xyz files into input representations
for machine learning algorithms.

For example, the Many-body tensor representation (MBTR).

# Python package dependencies
	1. multiprocessing
	2. ase
	3. scipy
	4. dscribe
	5. pandas

# Usage instructions
	1. `git clone https://github.com/kunalghosh/Multi_Fidelity_Prediction_GP.git`
	2. `cd Multi_Fidelity_Prediction_GP/task0`
	3. `sh gen_onehost.sh`
		* Downloads the dataset
		* Extracts XYZ file
		* Save the MBTR as scipy.sparse `lil_matrix`
		* MBTR data saved in `build_mbtr/mbtr.npz`
	4. Load data as follows
		```python
		from scipy.sparse import load_npz
		data = load_npz('build_mbtr/mbtr.npz')
		```
