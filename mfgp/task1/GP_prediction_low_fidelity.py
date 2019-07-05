import sys

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.model_selection import train_test_split

from utils import get_level

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

# Setup the GP model
noise = 0.4
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)

# Fit the model
gpr.fit(X_train.toarray(), y_train)

# predict
mu_s, std_s = gpr.predict(X_test.toarray(), return_std=True)

# save data
np.savez("data.npz",
         ids_train=ids_train,
         ids_test=ids_test,
         homo_lowfid_train=y_train,
         homo_lowfid_test=y_test,
         test_pred_mean=mu_s,
         test_pred_std=std_s)

# save the sparse matrices
save_npz("train_mbtr_sparsematrix.npz", X_train.tocsr())
save_npz("test_mbtr_sparsematrix.npz", X_test.tocsr())
