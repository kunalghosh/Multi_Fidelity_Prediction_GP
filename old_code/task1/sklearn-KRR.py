import sys
import pdb

import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from scipy import sparse
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gp_pytorch import ExactGPModel, model_fit, model_predict
from mfgp.utils import get_level
import torch.nn as nn

from sklearn.kernel_ridge import KernelRidge

np.random.seed(1)

if torch.cuda.is_available():
    useGPU=True
else:
    useGPU=False

# Load Data
mbtr_path = sys.argv[1]
json_path = sys.argv[2]

mbtr_data = load_npz(mbtr_path)
df_62k = pd.read_json(json_path, orient='split')

homo_lowfid = df_62k.apply(lambda row: get_level(
    row, level_type='HOMO', subset='PBE+vdW_vacuum'),
                           axis=1).to_numpy()

idxs = np.arange(len(homo_lowfid))
# Compute training and test splits
ids_train, ids_test = train_test_split(idxs, train_size=2000, random_state=0)
ids_test, _ = train_test_split(ids_test, train_size=2000, random_state=0)
mbtr_data.data = np.nan_to_num(mbtr_data.data)
X_train, X_test = mbtr_data[ids_train, :], mbtr_data[ids_test, :]
y_train, y_test = homo_lowfid[ids_train], homo_lowfid[ids_test]

X_train, X_test = X_train.toarray(), X_test.toarray()

# x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
# y_mean, y_std = y_train.mean(), y_train.std()

# X_train = (X_train-x_mean)/x_std
# X_test = (X_test-x_mean)/x_std
# y_train = (y_train-y_mean)/y_std

# print(x_mean.shape)

krr = KernelRidge(kernel='rbf',alpha=1e-5,gamma=1e-8)
krr.fit(X_train, y_train)
print(krr.score(X_test, y_test))
pred = krr.predict(X_test) 
# pred = pred * y_std + y_mean
print(pred)
print(f"error = {np.mean(np.abs(pred-y_test))}")
np.savez("pred.npz", pred=pred)
