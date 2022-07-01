import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


matplotlib.rc('font', **font)

def get_MAE(filename):
	MAE_list = []
	with open(filename) as f:
		for line in f.readlines():
			if line.startswith("MAE"):
				MAE_list.append(float(line.split()[1]))
	return MAE_list


file_1kbatches_stratD = "../test-D-1k"
file_expbatches_stratD = "../test-D"

x_label = "Batch size (x1k)"
y_label = "Test set MAE (eV)"

out_filename = "MAE.pdf"

mae_list_1k_D = get_MAE(file_1kbatches_stratD)
mae_list_exp_D = get_MAE(file_expbatches_stratD)

x_1k_d = np.arange(1, len(mae_list_1k_D)+1) # 1,2,3,....,12
x_exp_d = 2**np.arange(len(mae_list_exp_D)) # 1, 2, 4, 8, 16

plt.scatter(x_1k_d, mae_list_1k_D) 
plt.plot(x_1k_d, mae_list_1k_D, label="1k batches") 
plt.scatter(x_exp_d, mae_list_exp_D) 
plt.plot(x_exp_d, mae_list_exp_D, label="exp batches") 
plt.grid()
plt.legend()
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.savefig(out_filename)
