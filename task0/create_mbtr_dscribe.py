##################################################
# Author : Annika Stuke (annika.stuke@aalto.fi)  #
##################################################

import multiprocessing

import ase.io

from scipy.sparse import lil_matrix, save_npz

# import sys
# sys.path.insert(0, '/wrk/astuke/DONOTREMOVE/mbtr/describe')

from dscribe.descriptors import MBTR
from dscribe.utils.stats import system_stats


def create(i_samples):
    """This is the function that is called by each process but with different
    parts of the data.
    """
    n_i_samples = len(i_samples)
    i_res = lil_matrix((n_i_samples, n_features))
    for i, i_sample in enumerate(i_samples):
        feat = mbtr_desc.create(i_sample)
        i_res[i, :] = feat
        print("{} %".format((i + 1) / n_i_samples * 100))
    return i_res


# Load the systems
ase_atoms = list(ase.io.iread("../../data/data.xyz", format="xyz"))

# Load in statistics from the dataset
stats = system_stats(ase_atoms)
atomic_numbers = stats["atomic_numbers"]
max_atomic_number = stats["max_atomic_number"]
min_atomic_number = stats["min_atomic_number"]
min_distance = stats["min_distance"]

decay_factor = 0.5
mbtr_desc = MBTR(atomic_numbers=atomic_numbers,
                 k=[1, 2, 3],
                 periodic=False,
                 grid={
                     "k1": {
                         "min": min_atomic_number,
                         "max": max_atomic_number,
                         "sigma": 0.2,
                         "n": 200,
                     },
                     "k2": {
                         "min": 0,
                         "max": 1 / min_distance,
                         "sigma": 0.02,
                         "n": 200,
                     },
                     "k3": {
                         "min": -1.0,
                         "max": 1.0,
                         "sigma": 0.09,
                         "n": 200,
                     }
                 },
                 weighting={
                     "k2": {
                         "function": "exponential",
                         "scale": decay_factor,
                         "cutoff": 1e-3
                     },
                     "k3": {
                         "function": "exponential",
                         "scale": decay_factor,
                         "cutoff": 1e-3
                     },
                 }
                 # flatten = False
                 )

# Split the data into roughly equivalent chunks for each process
n_proc = 24  # How many processes are spawned
k, m = divmod(len(ase_atoms), n_proc)
atoms_split = (ase_atoms[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
               for i in range(n_proc))
n_features = int(mbtr_desc.get_number_of_features())

# Initialize a pool of processes, and tell each process in the pool to
# handle a different part of the data
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
