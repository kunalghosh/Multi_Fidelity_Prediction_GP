"""
Plot the strategy D vs B. Here D is the best strategy so far and B is 
the worst strategy. We see that D is able to achive the same accuracy
with far fewer datapoints compared to B. Initially I was plotting D vs A
but Jari gave a feedback that the improvement is not much so better to plot
against a strategy where the improvement is much more significant.
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
		plt.errorbar(batch_sizes, mean, std, capsize=lines['linewidth'])
		plt.scatter(batch_sizes, mean, label=labels[idx])

# draw a horizontal line with x = batch_sizes and y = accuracy of largest dataset with random
a_idx = labels.index('A')
plt.hlines(means[a_idx][-2], 0, batch_sizes[-1], linestyle="dashed", zorder=3) 
plt.legend()
plt.xticks(np.arange(0,18, step=2)) 
plt.xlabel("Dataset size in 1000s")
plt.ylabel("Mean absolute error (eV)")
plt.grid()
plt.tight_layout()
plt.savefig("Plot_B_and_D.pdf")
plt.close()
plt.clf()

# -------------------- Plot B and D with curve fits ---------------------

coeffs = { 'A' : None, 'D' : None }
mean_dict = {'A' : None, 'D' : None}

for idx, (mean, std) in enumerate(zip(means,stds)):
    if labels[idx] in ['A', 'D']:
        """
        Only plot strategy A and D.
        """
        print(f"Data for strategy {labels[idx]}")

        results = curve_fit(lambda x,a,b,c,d: d + (a-d)/(1 + (x/c)**b), batch_sizes, mean, p0=[0.5, 0.5, 0.5, 0.5], bounds=(-1, [3., 1., 2, 2]))
        a, b, c, d= results[0]
        coeffs[labels[idx]] = results[0]
        mean_dict[labels[idx]] = mean
        print(a, b, c, d)
        x = np.arange(1, 17)
        y = d + (a-d)/(1 + (x/c)**b) 

        # results = curve_fit(lambda x,a,b: a*x**b, batch_sizes, mean, p0=[0.5, -0.4])#, bounds=(0, [3., 1., 0.5]))
        # a, b= results[0]
        # print(a, b)
        # x = np.arange(1, 17)
        # y = a*x**b 

        # results = curve_fit(lambda x,a,b,c,d: a*np.exp(b*x**d)+c, batch_sizes, mean, p0=[0.2, 0.4, 0.2, 2])#, bounds=(0, [3., 1., 0.5]))
        # a, b, c, d = results[0]
        # print(a, b, c, d)
        # x = np.arange(1, 17)
        # y = a * np.exp(b * x**d) + c

        # results = curve_fit(lambda x,a,b,c,d: a*x**3 + b*x**2 + c*x + d, batch_sizes, mean)#, p0=[2, -1, 0.5])#, bounds=(0, [3., 1., 0.5]))
        # a, b, c, d = results[0]
        # print(a, b, c, d)
        # x = np.arange(1, 16)
        # y = a*x**3 + b*x**2 + c*x + d
        plt.plot(x, y)
        plt.scatter(batch_sizes, mean, label=labels[idx])

#-------------- Draw lines indicating for which value of x, D achieves same accuracy (y-value) as B ----------------
# for each mean in B, find the corresponding 'x' in D
xticks = batch_sizes.copy()
data_saving = []
a,b,c,d = coeffs['D']
for y, batch_size in zip(mean_dict['A'], batch_sizes):
    x = c * (-1 + (a-d)/(y-d))**(1./b)
    xticks.append(x)
    data_saving.append(batch_size - x)
    print(batch_size, x)
    # vertical line at x
    # plt.axvline(x=x, color='k', linestyle="--")
    plt.plot((x, x), (0, y), color='k', linestyle="--")
    plt.hlines(y, x, batch_size, color='k', linestyle="--")

xticks.sort()
print(xticks)
plt.legend()
xticks_str = []
for idx, x in enumerate(xticks):
    if x in batch_sizes:
        xticks_str.append(" ")
    else:
        xticks_str.append("%.2f" % x)

print(xticks_str)

plt.xscale("log")
plt.tick_params(axis='x', which='minor')
plt.xticks(ticks=sorted(xticks), labels=xticks_str)
plt.xticks(rotation=45)
plt.xlabel("Dataset size (x10^3)")
plt.ylabel("Mean absolute error (eV)")
plt.grid()
plt.tight_layout()
plt.savefig("Plot_A_and_D_curvefit.pdf")
plt.close()
plt.clf()

# --------------- data for this plot is generated in the previous section ----------------------

data_saving = np.array(data_saving)
batch_sizes = np.array(batch_sizes)
savings_in_percent = 100 * data_saving / batch_sizes
print(f"datasavings {data_saving}")
plt.plot(batch_sizes, savings_in_percent, color="k")
plt.scatter(batch_sizes, savings_in_percent, color="k", label="Datasaving (D vs A) in percent")
plt.xscale("linear")
plt.legend()
plt.xticks(batch_sizes)
plt.xlabel("Data size (x10^3)")
plt.ylabel("Datasaving in percent")
plt.grid()
plt.tight_layout()
plt.savefig("Plot_AvsD_savings.pdf")
plt.close()
plt.clf()
