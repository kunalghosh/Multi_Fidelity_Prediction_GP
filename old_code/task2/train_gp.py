# Run `train_gp.py train_idxs.npz` : Trains an exact GP on the entire list of 
# indices in `train_idxs.npz`. If there are multiple rows it is flattened and
# the training is done on the entire dataset. 
# This also generates a new file `predictive_means_and_vars_iter{i}.npy`

import numpy as np
np.random.seed(1)


