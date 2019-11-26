# Run `init_train_idxs.py <int: dataset size> <int: initial training set size>`:
#  Creates a `train_idxs.npz` file with the initial set of training indices. 
# e.g `python init_train_idxs.py 64000 1000`

import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Script to initialize trainin, validation and test indices for active learning of molecule dataset.")
parser.add_argument("dataset_size", type=int, help="""Total number of elements in the dataset (S), if you enter less than the total number (N < S), then the first N elements are only used to generate the train validation and test set.""")
parser.add_argument("initial_trainset_size", type=int, help="""Initial number of elements in the training set.""")
parser.add_argument("--valid_size", type=int, default=500, help="""Size of the validation set (exact number of elements, not float value indicating the portion of train set)""")
parser.add_argument("--test_size", type=int, default=1000, help="""Test set size, first the test set is taken from the entire dataset then the train and validation are computed from the rest.""")
parser.add_argument("--filename", type=str, default="train_idxs", help="""Filename where to store the indices, WITHOUT the extension.""")


args = parser.parse_args()

dataset_size = args.dataset_size
init_trainset_size = args.initial_trainset_size
validation_set_size = args.valid_size # usually training set is much larger so 500 is reasonable
test_set_size = args.test_size
filename = args.filename + ".npz"

np.random.seed(1)
# Because the test set is computed first with the same random seed, 
# they would be the same everytime this script is called
# (given that the test set size is kept constant).
test_idxs, remaining_idxs = train_test_split(range(dataset_size), train_size = test_set_size, random_state=0)
train_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = init_trainset_size, random_state=0)
valid_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = validation_set_size, random_state=0)

# save the values in train_idxs.npy
np.savez(filename, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs)
