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
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel, Matern

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
ids_train, ids_test = train_test_split(idxs, train_size=1000, random_state=0)
ids_test, _ = train_test_split(ids_test, train_size=1000, random_state=0)
mbtr_data.data = np.nan_to_num(mbtr_data.data)
X_train, X_test = mbtr_data[ids_train, :], mbtr_data[ids_test, :]
y_train, y_test = homo_lowfid[ids_train], homo_lowfid[ids_test]

X_train, X_test = X_train.toarray(), X_test.toarray()

x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
x_std = x_std + 1 # to ensure we don't divide by zero. if std is 0
print(f"mean : {np.min(x_mean)}, {np.min(x_mean)}")
print(f"std  : {np.min(x_std)}, {np.min(x_std)}")
y_mean, y_std = y_train.mean(), y_train.std()

normalize_y = False
# X_train = (X_train-x_mean)/x_std
# X_test = (X_test-x_mean)/x_std
if normalize_y:
    y_train = (y_train-y_mean)/y_std

# print(x_mean.shape)

# kernel = ConstantKernel(1, constant_value_bounds=(1e-8, 1e2)) * RBF(length_scale=1e-2, length_scale_bounds=(1e-10, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
n_features = X_train.shape[-1]
print(f"N features : {n_features}")
# kernel = RBF(length_scale=[7071], length_scale_bounds=(1e7, 1e8)) # best result.
kernel = RBF(length_scale=[1e7]*n_features, length_scale_bounds=(1e7, 1e8)) # ARD Kernel
# kernel = ConstantKernel(1, constant_value_bounds=(1e-5, 1e1)) * Matern(1e-16, length_scale_bounds=(1e-16, 1), nu=2.5)

## Compute the kernel on train and test set
## with current (kernel) parameters. It shoud
## be close to 1 indicating that the train and test
## points are similar as measured by the kernel.


# ker_val = kernel(X_train, X_test)
# print(ker_val.shape)
# print(np.max(ker_val))
# print(np.min(ker_val))
# print(f"kernel train-test, {ker_val}")

# def optimizer_func(obj_func, initial_theta, bounds):
# 	x, f, d = fmin_l_bfgs_b(func=obj_func, x0=initial_theta, bounds=bounds, maxiter=15000)
# 	return x, f

# gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, optimizer=optimizer_func)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y)
# print(f"Before : params = {gpr.get_params()}")
gpr.fit(X_train, y_train)
print("Training done !")
# print(f"After : params = {gpr.get_params()}")

print(gpr.score(X_test, y_test))
# pred_train = gpr.predict(X_train, return_std=True) 
# print(f"Predict train.....{pred_train}")
# print(f"predict train mae {np.mean(np.abs(pred_train[0] - y_train))}")

pred = gpr.predict(X_test, return_std=True) 
print(pred)
pred = pred[0] # just the means
if normalize_y:
    pred = pred * y_std + y_mean
print(pred)
print(f"mae error = {np.mean(np.abs(pred-y_test))}")
print(f"Kernel params = {gpr.kernel_.get_params()}")
# print(f"kernel hyperparams = {kernel.hyperparameters}")
print(f"kernel theta = {kernel.theta}")
np.savez("pred.npz", pred=pred)

## Similar kernel computation as before.
## again the train and test points must
## be similar as seen by the kernel.

# ker_val = kernel(X_train, X_test)
# print(ker_val.shape)
# print(np.max(ker_val))
# print(np.min(ker_val))
# print(f"kernel train-test, {ker_val}")


