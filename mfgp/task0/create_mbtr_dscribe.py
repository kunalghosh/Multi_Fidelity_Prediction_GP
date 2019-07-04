##################################################
# Author : Kunal Ghosh (kunal.ghosh@aalto.fi)    #
# Author : Annika Stuke (annika.stuke@aalto.fi)  #
##################################################

import multiprocessing

import ase.io

from scipy.sparse import lil_matrix, save_npz, load_npz

import sys
# sys.path.insert(0, '/wrk/astuke/DONOTREMOVE/mbtr/describe')

from dscribe.utils.stats import system_stats
from mbtr_utils import create_mbtr, make_mbtr_desc

# script accepts filenames as input
# xyz_file is the input file with xyz data
# mbtr_file is the output file with mbtr as scipy sparse matrix (lil_matrix)
xyz_file = sys.argv[1]
# mbtr_file = sys.argv[2]

# # Load the mbtr file
# try:
#     mbtr_precomputed = load_npz(mbtr_file)
# except FileNotFoundError as e:
#     print(e)
#     mbtr_precomputed = []

# Load the systems
ase_atoms = list(ase.io.iread(xyz_file, format="xyz"))

# Load in statistics from the dataset
stats = system_stats(ase_atoms)
atomic_numbers = stats["atomic_numbers"]
max_atomic_number = stats["max_atomic_number"]
min_atomic_number = stats["min_atomic_number"]
min_distance = stats["min_distance"]

decay_factor = 0.5

mbtr_desc = make_mbtr_desc(atomic_numbers, max_atomic_number,
                           min_atomic_number, min_distance, decay_factor)

# Split the data into roughly equivalent chunks for each process
n_proc = 24  # How many processes are spawned
k, m = divmod(len(ase_atoms), n_proc)
atoms_split = (ase_atoms[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
               for i in range(n_proc))
n_features = int(mbtr_desc.get_number_of_features())


# Initialize a pool of processes, and tell each process in the pool to
# handle a different part of the data
def create(indices):
    return create_mbtr(mbtr_desc, n_features, indices)


pool = multiprocessing.Pool(processes=n_proc)
res = pool.map(create, atoms_split)  # pool.map keeps the order

# Save results
n_samples = len(ase_atoms)
mbtr_list = lil_matrix((n_samples, n_features))

i_sample = 0
for i, i_res in enumerate(res):
    i_n_samples = i_res.shape[0]
    mbtr_list[i_sample:i_sample + i_n_samples, :] = i_res
    i_sample += i_n_samples

# Saves the descriptors as a sparse matrix
mbtr_list = mbtr_list.tocsr()
save_npz("mbtr.npz", mbtr_list)
