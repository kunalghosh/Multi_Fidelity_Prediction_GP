"""
For the re-run of Shingo's code (Random acquisition strategy), 
plot the time taken to run each batch.
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

lines = {"linewidth": 3.0}

matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)

# file with timings
shingo_Dfile = "../test-D"

batch_sizes = [1, 2, 4, 8, 16]
string_this_step ="time for this step"
string_overall_time = "time all"

def get_times(filepath):
	"""
	Arguments:
		filepath (String) : Accepts path to data dump file.
	
	Returns:
		step_times (List) : 
			List of parsed out times (in seconds) for each step.
		total_times (List) :
			Cumulative times (in seconds) for each step.
	""" 
	step_times = []
	total_times = []

	with open(filepath) as f:
		for line in f.readlines():
			if line.startswith(string_this_step):
				match = re.search("[0-9]+\.[0-9]+", line)
				step_times.append(float(match.group()))
			
			if line.startswith(string_overall_time):
				match = re.search("[0-9]+\.[0-9]+", line)
				total_times.append(float(match.group()))
	return step_times, total_times


step_times, total_times = get_times(shingo_Dfile)
print(f"exp batches {step_times}")
print(f"total times {total_times}")
total_times = np.array(total_times)/3600
plt.scatter(batch_sizes, total_times[:-1]) 
plt.plot(batch_sizes, total_times[:-1], label="exp  batches (Singo D)") 
print(total_times[-3])
plt.hlines(total_times[-3], batch_sizes[0], batch_sizes[-1], linestyles="dashed")
# plt.yscale("log")
# plt.xscale("log")

# plot 1k batches
step_times, total_times = get_times("../test-D-1k")
print(f"1k batches {step_times}")
print(f"total times {total_times}")
if len(total_times) == 16:
	total_times = total_times[:-1]
batch_sizes = np.arange(1, len(total_times)+1)

total_times = np.array(total_times)/3600
plt.scatter(batch_sizes, total_times) 
plt.plot(batch_sizes, total_times, label="1k batches") 
# plt.yscale("log")
# plt.xscale("log")
plt.legend()
plt.xlabel("Dataset size in 1000s")
plt.ylabel("Time in hours")
plt.grid()
plt.tight_layout()
plt.savefig("Strategy_D_shingo.pdf")
plt.close()


# fit exponential curve 
results = curve_fit(lambda t,a,b: a*np.exp(b*t), batch_sizes,  total_times, p0=(3.0, 0.1))


# plot exponential fit
a, b = results[0]
print(a,b)
x = np.arange(1, 33)
y = a * np.exp(x * b)
plt.scatter(x, y)
plt.plot(x, y, label="exp fitted 1k batches")
plt.yscale("log")
# plt.xscale("log")

# plot previous result
plt.scatter(batch_sizes, total_times) 
plt.plot(batch_sizes, total_times, label="1k batches") 
plt.yscale("log")
# plt.xscale("log")

plt.legend()
plt.xlabel("Dataset size in 1000s")
plt.ylabel("Time in hours")
plt.grid()
plt.tight_layout()
plt.savefig("Curve_fit.pdf")
plt.close()
