# Run `init_train_idxs.py <int: dataset size> <int: initial training set size>`:
#  Creates a `train_idxs.npz` file with the initial set of training indices. 
# e.g `python init_train_idxs.py 64000 1000`

import sys
import numpy as np
from sklearn.model_selection import train_test_split

dataset_size = int(sys.argv[1])
init_trainset_size = int(sys.argv[2])
validation_set_size = 500 # usually training set is much larger so 500 is reasonable

np.random.seed(1)

train_idxs, remaining_idxs = train_test_split(range(dataset_size), train_size = init_trainset_size, random_state=0)
valid_idxs, test_idxs = train_test_split(remaining_idxs, train_size = 500, random_state=0)

# save the values in train_idxs.npy
np.savez("train_idxs.npz", train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs)
