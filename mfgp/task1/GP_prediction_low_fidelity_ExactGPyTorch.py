import sys

import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split

from gp_pytorch import ExactGPModel, model_fit, model_predict
from mfgp.utils import get_level

np.random.seed(1)

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
ids_train, ids_test = train_test_split(idxs, test_size=0.3, random_state=0)
X_train, X_test = mbtr_data[ids_train, :], mbtr_data[ids_test, :]
y_train, y_test = homo_lowfid[ids_train], homo_lowfid[ids_test]

# convert data to tensor
x_train_tensor = torch.tensor(X_train.toarray())
x_test_tensor = torch.tensor(X_test.toarray())
y_train_tensor = torch.tensor(y_train)
# Setup the GP model
likelihood = GaussianLikelihood()
gpr = ExactGPModel(x_train_tensor, y_train_tensor, likelihood)

# Fit the model
model_fit(model=gpr,
          likelihood=likelihood,
          x_train=x_train_tensor,
          y_train=y_train_tensor)

# predict
mu_s = model_predict(
    model=gpr,
    likelihood=likelihood,
    x_test=x_test_tensor,
)

# save data
np.savez(
    "gpytorch_data.npz",
    ids_train=ids_train,
    ids_test=ids_test,
    homo_lowfid_train=y_train,
    homo_lowfid_test=y_test,
    test_pred_mean=mu_s,
)

# save the sparse matrices
save_npz("gpytorch_train_mbtr_sparsematrix.npz", X_train.tocsr())
save_npz("gpytorch_test_mbtr_sparsematrix.npz", X_test.tocsr())
