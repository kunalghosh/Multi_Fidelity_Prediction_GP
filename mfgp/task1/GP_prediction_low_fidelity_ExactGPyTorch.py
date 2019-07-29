import sys

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
ids_train, ids_test = train_test_split(idxs, train_size=100, random_state=0)
ids_test, _ = train_test_split(ids_test, train_size=100, random_state=0)
mbtr_data.data = np.nan_to_num(mbtr_data.data)
X_train, X_test = mbtr_data[ids_train, :], mbtr_data[ids_test, :]
y_train, y_test = homo_lowfid[ids_train], homo_lowfid[ids_test]

# convert data to tensor
x_train_tensor = torch.Tensor(X_train.toarray().squeeze())
x_test_tensor = torch.Tensor(X_test.toarray().squeeze())
y_train_tensor = torch.Tensor(y_train)

if useGPU:
    x_train_tensor = x_train_tensor.cuda()
    x_test_tensor = x_test_tensor.cuda()
    y_train_tensor = y_train_tensor.cuda()

print(x_train_tensor.shape)
print(x_test_tensor.shape)

_train_mean = x_train_tensor.mean()
_train_std  = x_train_tensor.std()

x_train_tensor = (x_train_tensor - _train_mean)/_train_std
x_test_tensor = (x_test_tensor - _train_mean)/_train_std
# Setup the GP model
likelihood = GaussianLikelihood()

# Initialize the inducing points using KMeans
M = 20 # Number of inducing points
kmeans = KMeans(n_clusters=M, random_state=0).fit(x_train_tensor)
z = torch.Tensor(kmeans.cluster_centers_)
gpr = ExactGPModel(x_train_tensor, y_train_tensor, likelihood, z)

if useGPU:
    gpr = gpr.cuda()

# Fit the model
model_fit(model=gpr,
          likelihood=likelihood,
          x_train=x_train_tensor,
          y_train=y_train_tensor,
          max_epochs=751, lr=0.1)

# predict
mu_s = model_predict(
    model=gpr,
    likelihood=likelihood,
    x_test=x_test_tensor,
)

# save model
statedict_filename = "model_state.pth"
sparsedata_filename = "gpytorch_sparse_data.npz"
homotrain_filename = "homo_data_train.npz"
homotest_filename = "homo_data_test.npz"
mbtrtrain_filename = "gpytorch_sparse_train_mbtr_sparsematrix.npz"
mbtrtest_filename = "gpytorch_sparse_test_mbtr_sparsematrix.npz"

prependCPU = lambda x: "cpu_" + x
prependGPU = lambda x: "gpu_" + x

if useGPU:
    statedict_filename  = prependGPU(statedict_filename)
    sparsedata_filename = prependGPU(sparsedata_filename)
    homotrain_filename  = prependGPU(homotrain_filename)
    homotest_filename   = prependGPU(homotest_filename)
    mbtrtrain_filename  = prependGPU(mbtrtrain_filename)
    mbtrtest_filename   = prependGPU(mbtrtest_filename)
else:
    statedict_filename  = prependCPU(statedict_filename)
    sparsedata_filename = prependCPU(sparsedata_filename)
    homotrain_filename  = prependCPU(homotrain_filename)
    homotest_filename   = prependCPU(homotest_filename)
    mbtrtrain_filename  = prependCPU(mbtrtrain_filename)
    mbtrtest_filename   = prependCPU(mbtrtest_filename)

torch.save(gpr.state_dict(), statedict_filename)

# save data
np.savez(
    sparsedata_filename,
    ids_train=ids_train,
    ids_test=ids_test,
    test_pred_mean=mu_s.loc,
    # y_train=y_train,
    # y_test=y_test
)

save_npz(homotrain_filename, sparse.csr_matrix(y_train))
save_npz(homotest_filename, sparse.csr_matrix(y_test))

# save the sparse matrices
save_npz(mbtrtrain_filename, sparse.lil_matrix(X_train).tocsr())
save_npz(mbtrtest_filename, sparse.lil_matrix(X_test).tocsr())
