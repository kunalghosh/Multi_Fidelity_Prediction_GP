# coding: utf-8
import sys
import numpy as np
from sklearn.externals import joblib
from scipy.sparse import load_npz
from sklearn.metrics import mean_absolute_error

iteration = int(sys.argv[1])
test_idxs = np.load(f"test-A-1k_{iteration}_full_idxs.npz")['test_idxs']

gp = joblib.load(f"test-A-1k_{iteration}_model.pkl")

mbtr_path = "/projappl/project_2000382/ghoshkun/data/AA/mbtr_k2.npz"
json_path = "/projappl/project_2000382/ghoshkun/data/AA/HOMO.txt"

mbtr_data = load_npz(mbtr_path)
homo_lowfid = np.loadtxt(json_path)

X_test = mbtr_data[test_idxs, :].toarray(); y_test = homo_lowfid[test_idxs]
mu_s, std_s = gp.predict(X_test, return_std=True)

MAE = np.array(mean_absolute_error(y_test, mu_s))
print(f"{iteration} {MAE}")
