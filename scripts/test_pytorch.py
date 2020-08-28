from aldc.models import GPytorchGPModel
import torch
import gpytorch
import numpy as np
from scipy.sparse import load_npz, save_npz
import os
from aldc.strategies import acq_fn
import json
from sklearn.model_selection import train_test_split

homo_lowfid = np.loadtxt("HOMO.txt")
mbtr_data = load_npz("mbtr_k2.npz")

random_seed = 12
test_set_size = 5000

def pre_rem_split(prediction_set_size, remaining_idxs, random_seed):
    if len(remaining_idxs) == prediction_set_size :
        prediction_idxs = remaining_idxs
        remaining_idxs = np.array([])
#        remaining_idxs = {}
    elif len(remaining_idxs) != prediction_set_size:
        prediction_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = prediction_set_size, random_state=random_seed)#, random_state=0)
    return prediction_idxs, remaining_idxs

#init first batch and test set
mbtr_data_size = mbtr_data.shape[0]
test_idxs, remaining_idxs = train_test_split(range(mbtr_data_size), train_size = test_set_size, random_state = random_seed)
batch_size = 4000
prediction_idxs, remaining_idxs = pre_rem_split(batch_size, remaining_idxs, random_seed)

#take mbtr_data corresponding to first bacth ids
X_train = mbtr_data[prediction_idxs, :]
train_x = X_train.toarray()
train_y  = homo_lowfid[prediction_idxs]

#take mbtr_data corresponding to test set
X_test = mbtr_data[test_idxs, :]
X_test = X_test.toarray()
y_test = homo_lowfid[test_idxs]

#train_x = load_npz("mbtr_train_set.npz")
#train_x = train_x.toarray()
#train_y = np.loadtxt("homo_lowfid_train_set.txt")
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = GPytorchGPModel(train_x, train_y, likelihood)
gp.set_params(20.0, 700.0)
gp.fit(train_x, train_y,X_test,y_test, torch_optimizer=torch.optim.Adam, lr=1.0, max_epochs=300, lr_step=1000, gamma=0.5)
