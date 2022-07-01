# coding: utf-8
import sys
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import mean_absolute_error

iteration = int(sys.argv[1])
strategy = (sys.argv[2]) # strategy A, B, C, D
batch_size = (sys.argv[3]) # batch size 1k,2k, ... EXP
dataset = sys.argv[4] # dataset one of AA, OE, QM9
mbtr_filename = {"AA":"mbtr_k2.npz",
    "OE":"mbtr_0.02.npz",
    "QM9":"mbtr_0.1.npz"}

print(f"iteration {iteration}"
        f"strategy {strategy}"
        f"batch size {batch_size}"
        f"dataset {dataset}"
        f"mbtr_filename mbtr_filename")

test_idxs = np.load(f"test-{strategy}-{batch_size}_{iteration}_full_idxs.npz")['test_idxs']

gp = joblib.load(f"test-{strategy}-{batch_size}_{iteration}_model.pkl")

mbtr_path = f"/projappl/project_2000382/ghoshkun/data/{dataset}/{mbtr_filename[dataset]}"
json_path = f"/projappl/project_2000382/ghoshkun/data/{dataset}/HOMO.txt"

mbtr_data = load_npz(mbtr_path)
homo_lowfid = np.loadtxt(json_path)

X_test = mbtr_data[test_idxs, :].toarray(); y_test = homo_lowfid[test_idxs]
mu_s, std_s = gp.predict(X_test, return_std=True)

MAE = np.array(mean_absolute_error(y_test, mu_s))
print(f"{iteration} {MAE}")
