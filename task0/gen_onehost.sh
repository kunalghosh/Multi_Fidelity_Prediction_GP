#! /usr/bin/sh -e
# script that is run on a Single Host, with Multiple Cores to generate MBTR from XYZ files.
# -e allows one to ignore failed commands with '|| true'

cd $WRKDIR || true

mkdir build_mbtr
cd build_mbtr
echo "Created and changed directory to a temp directory."

# download and extract the JSON file containing the data
curl -C - -Lo df_full_split_ids_with_smiles_v15.json.tar.gz https://users.aalto.fi/ghoshk1/df_full_split_ids_with_smiles_v15.json.tar.gz || true
echo "Downloaded the data"

tar -xvzf df_full_split_ids_with_smiles_v15.json.tar.gz
echo "Extracted the tar.gz file"

# setup conda environment for subsequent python scripts
module load anaconda3 || true
echo "Loaded anaconda module"

conda create -p ./myenv numpy scipy matplotlib pandas
echo "Created the Conda environment"

source activate ./myenv
echo "Activated the conda environment"

pip install dscribe gpytorch ase
conda install pytorch -c pytorch
echo "Installed additional packages."

# extract the xyz data as csv from the json file
python ../create_xyz.py df_full_split_ids_with_smiles_v15.json data.csv
echo "Extracted XYZ data into a CSV file."

# clean csv file to generate the xyz file
cat data.csv | grep -v '^"$' | tr -d '"'  > data.xyz
echo "Converted CSV to XYZ file."

# generate MBTR from XYZ file
python ../create_mbtr_dscribe.py data.xyz new_mbtrfile.npz
echo "Generated MBTR file."

# # delete generated data
# rm data.csv data.xyz
# echo "Deleted intermediate files"

