"""
Plot the strategy D vs A. Here D is the best strategy so far and A is 
the random strategy. We see that D is able to achive the same accuracy
with far fewer datapoints compared to A.
"""
import re
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# formatting plots

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 15}

lines = {"linewidth": 4}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)

# file with timings

batch_sizes = [1, 2, 4, 8, 16]
labels = "A B C D E F G".split(" ")
labs = [[_] for _ in labels]

stds = [[0.016656652, 0.009579919, 0.007564725, 0.008139959, 0.001847635],
	 [0.021067, 0.011635637, 0.012508936, 0.019852679, 0.009236511],
	 [0.010166611, 0.014369837, 0.005485079, 0.002348969, 0.001101183],
	 [0.013155269, 0.010881134, 0.008067763, 0.003085855, 0.002080145],
	 [0.022870525, 0.010037362, 0.005520057, 0.004963253, 0.002371183],
	 [0.029183907, 0.01674064, 0.013705215, 0.005953054, 0.000939363],
	 [0.022870525, 0.010037362, 0.005520057, 0.004963253, 0.002371183]]

means = [[0.527572612, 0.395494212, 0.305729616, 0.230183849, 0.170217741],
	[0.538030813, 0.437973463, 0.323704362, 0.254709336, 0.164451888],
	[0.533625252, 0.389543845, 0.304730031, 0.229775939, 0.165909217],
	[0.52634072, 0.367366228, 0.269792825, 0.203872432, 0.147824629],
	[0.532987113, 0.382061848, 0.278853979, 0.200607586, 0.145790025],
	[0.530769344, 0.385524992, 0.281062877, 0.203867492, 0.151144159],
	[0.532987113, 0.382061848, 0.278853979, 0.200607586, 0.145790025]]

for idx, (mean, std) in enumerate(zip(means,stds)):
	if labels[idx] in ['A', 'D']:
		"""
		Only plot strategy A and D.
		"""
		plt.errorbar(batch_sizes, mean, std, label=labels[idx], capsize=lines['linewidth'])
		plt.scatter(batch_sizes, mean)

# draw a horizontal line with x = batch_sizes and y = accuracy of largest dataset with random
a_idx = labels.index('A')
plt.hlines(means[a_idx][-1] ,batch_sizes[0], batch_sizes[-1], linestyle="dashed", zorder=3) 
plt.legend()
plt.xticks(np.arange(0,18, step=2)) 
plt.xlabel("Dataset size in 1000s")
plt.ylabel("Mean absolute error (eV)")
plt.grid()
plt.tight_layout()
plt.savefig("Plot_A_and_D.pdf")
plt.close()
plt.clf()
