# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook is used to make plots. The notebook is converted to a plain python text file using [jupytext](https://jupytext.readthedocs.io/en/latest/)

# %%
import os
import numpy as np
import pandas as pd

# %%
os.chdir("/projappl/project_2000382/ghoshkun/code/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs")
# print(os.getcwd())
# # !module load python-data/3.7.6-1
# # !python -m pip install jupytext
# !python -m jupytext --sync MakePlots.ipynb

# %% [markdown]
# # Energy (x) vs Uncertainty (y) plot
# In this plot we load the python model for each batch in run 1 of a given Dataset/Strategy/Batches combo (e.g. AA_D_EXP ) and make a plot which shows where was the uncertainty reduced the most. The data for this plot is present in `Puhti` under `/scratch/project_2000382/ghoshkun/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs`

# %%
import os
os.getcwd()

# %% [markdown]
# Set the current working directory to where the data exists

# %%
os.chdir("/projappl/project_2000382/ghoshkun/code/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs")
assert os.getcwd() == "/projappl/project_2000382/ghoshkun/code/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs"

# %% [markdown]
# # Plot A vs D

# %%
from matplotlib import pyplot as plt
import matplotlib as mpl
# %matplotlib inline

def set_mpl_params(matplotlib):
    plt.figure(figsize=(6,4), dpi=200)
    # formatting plots
    font = {'family' : 'monospace', 'size'   : 15, 'sans-serif':'Nimbus'}
    lines = {"linewidth": 4}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)
    
def get_labels():
    labels = "A B C D E F G".split(" ")
    labs = [[_] for _ in labels]
    return labs, labels


# %%
def get_means_stds_batchsize(aa_a, aa_d):
    assert (aa_a.batch_size == aa_d.batch_size).all(), "batch_size column must be the same, in the two files passed in the argument."

    # batch_sizes = np.arange(0,17,batch_size) + 1
    batch_sizes = np.cumsum(aa_a.batch_size.to_numpy()) // 1000

    aa_a_means = aa_a.mean_vals.to_numpy()
    aa_d_means = aa_d.mean_vals.to_numpy()

    aa_a_stds = aa_a.std_vals.to_numpy()
    aa_d_stds = aa_d.std_vals.to_numpy()

    stds = [aa_a_stds,
         [],
         [],
         aa_d_stds,
         [],
         [],
         []]

    means = [aa_a_means,
         [],
         [],
         aa_d_means,
         [],
         [],
         []]
    return means, stds, batch_sizes


# %%
def plot_strategy_a_vs_d(dataset_name, means, stds, labels):
    for idx, (mean, std) in enumerate(zip(means,stds)):
        if labels[idx] in ['A', 'D']:
            """
            Only plot strategy A and D.
            """
            _ = plt.errorbar(batch_sizes, mean, std) #, capsize=lines['linewidth'])
            _ = plt.scatter(batch_sizes, mean, label=labels[idx])
    plt.grid()
    plt.xlabel("Dataset size in 1000s")
    plt.ylabel("MAE (eV)")
    plt.gca().tick_params(axis='x', which='minor', bottom=True)
    plt.xscale("linear")
    #plt.xlim((0, 33))
    plt.ylim((0,0.6))
    # plt.xscale("log", basex=2)
    #     sformatter = mpl.ticker.ScalarFormatter()
    #     sformatter.set_scientific(False)
    formatter = mpl.ticker.FuncFormatter(lambda y, _: '{:0.0f}'.format(y))
    # use plt.gca() whenever there is a need for axis.<something>
    plt.gca().xaxis.set_major_formatter(formatter)
    
    #plt.xticks(range(0, 33))
    if dataset_name:
        plt.title(f"{dataset_name} - Strategy A vs D")


# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - AA_A_1k.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - AA_D_1k.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = None, means=means, stds=stds, labels=labels)    

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - QM9_A_EXP_physdays.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - QM9_D_EXP_physdays.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = None, means=means, stds=stds, labels=labels)    

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - AA_A_2k.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - AA_D_2k.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = "AA 2k", means=means, stds=stds, labels=labels)    

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - AA_A_4k.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - AA_D_4k.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = "AA 4k", means=means, stds=stds, labels=labels) 

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - AA_A_EXP_old.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - AA_D_EXP_old.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
print(batch_sizes)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = None, means=means, stds=stds, labels=labels)    

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - QM9_A_EXP_old.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - QM9_D_EXP_old.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
print(batch_sizes)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = None, means=means, stds=stds, labels=labels)    

# %%
aa_a = pd.read_csv("csv_files/Active_learning_results - QM9_A_EXP.csv")
aa_d = pd.read_csv("csv_files/Active_learning_results - QM9_D_EXP.csv")

means, stds, batch_sizes = get_means_stds_batchsize(aa_a,aa_d)
print(batch_sizes)
set_mpl_params(matplotlib)
labs, labels = get_labels()
plot_strategy_a_vs_d(dataset_name = None, means=means, stds=stds, labels=labels)    

# %% [markdown]
# TODO : For the plot with the new data, save the means and standard deviation.

# %% [markdown]
# # Data savings plot
#
# The code is based on the original implementation in 

# %%
from scipy.optimize import curve_fit
import numpy as np

def fit_curve_to_data(means, stds, labels, batch_sizes):
     # np.random.seed(0)
    coeffs = { 'A' : None, 'D' : None }
    mean_dict = {'A' : None, 'D' : None}
    for idx, (mean, std) in enumerate(zip(means,stds)):
        if labels[idx] in ['A', 'D']:
            """
            Only plot strategy A and D.
            """
#             results = curve_fit(lambda x,a,b,c,d: d + (a-d)/(1 + (x/c)**b), batch_sizes, mean, p0=[0.5, 0.5, 0.5, 0.5], bounds=(-1, [3., 1., 2, 2]), maxfev=10000)
            results = curve_fit(lambda x,a,b,c,d: d + (a-d)/(1 + (x/c)**b), batch_sizes, mean, p0=[0.5, 0.5, 0.5, 0.5], bounds=(-1, [1., 1., 2, 2]), maxfev=10000)
            a, b, c, d= results[0]
            coeffs[labels[idx]] = results[0]
            mean_dict[labels[idx]] = mean

            x = batch_sizes # np.arange(0, 17, batch_size)
            y = d + (a-d)/(1 + (x/c)**b)

    return coeffs, mean_dict


# fit_curve_to_data(means, stds, labels, batch_sizes)

# %%
def plot_strategy_a_vs_d(dataset_name, means, stds, labels):
    for idx, (mean, std) in enumerate(zip(means,stds)):
        if labels[idx] in ['A', 'D']:
            """
            Only plot strategy A and D.
            """
            _ = plt.errorbar(batch_sizes, mean, std)#, capsize=lines['linewidth'])
            _ = plt.scatter(batch_sizes, mean, label=labels[idx])
    plt.grid()
    plt.xlabel("Dataset size in 1000s")
    plt.ylabel("MAE (eV)")
    plt.xlim((0, 18))
    plt.xticks(range(0, 17, 2))
    plt.title(f"{dataset_name} - Strategy A vs D")
    
# plot_strategy_a_vs_d(dataset_name = "AA", means=means, stds=stds)    


# %%
def get_data_savings(mean_dict, coeffs, batch_size, batch_sizes):
    # xticks = [_ for _ in range(0,17,batch_size)]
    data_saving = []
    a,b,c,d = coeffs['D']
    for y, batch_size in zip(mean_dict['A'], batch_sizes):
        x = c * (-1 + (a-d)/(y-d))**(1./b)
        # xticks.append(x)
        data_saving.append(batch_size - x)

#     xticks.sort()
#     xticks_str = []
#     for idx, x in enumerate(xticks):
#         if x in batch_sizes:
#             xticks_str.append(" ")
#         else:
#             xticks_str.append("%.2f" % x)
    return data_saving

# get_data_savings(mean_dict, coeffs, batch_size, batch_sizes)


# %%
import os
import pandas as pd
import numpy as np
import matplotlib
# %matplotlib inline
from matplotlib import pyplot as plt

def plot_data_savings(strategy_a_file, strategy_d_file, batch_size, percentage=False):
    aa_a = pd.read_csv(strategy_a_file)
    aa_d = pd.read_csv(strategy_d_file)

    set_mpl_params(matplotlib)
    
    labs, labels = get_labels()
    
    means, stds, batch_sizes = get_means_stds_batchsize(aa_a, aa_d)
    # print(means[0], means[3], batch_sizes)
    
    coeffs, mean_dict = fit_curve_to_data(means, stds, labels, batch_sizes)
    # print(coeffs)
    data_saving = get_data_savings(mean_dict, coeffs, batch_size, batch_sizes)
    
    data_saving = np.array(data_saving)
    batch_sizes = np.array(batch_sizes)

    print(data_saving)
    if percentage:
        data_saving = 100 * data_saving / (batch_sizes)
        
    plt.plot(batch_sizes, data_saving, color="k")
    plt.scatter(batch_sizes, data_saving, color="k")#, label="Datasaving (D vs A) in percent")
    plt.xscale("linear")
    # plt.legend()
    plt.xticks(batch_sizes)
    plt.xlabel("Dataset size in 1000s")
    plt.ylabel("Datasaving in percent")
    plt.grid()
    return data_saving, batch_sizes
    
os.chdir("/projappl/project_2000382/ghoshkun/code/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs")    
plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - AA_A_4k.csv",
                  strategy_d_file = "csv_files/Active_learning_results - AA_D_4k.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc

# %%
plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - QM9_A_EXP_physdays.csv",
                  strategy_d_file = "csv_files/Active_learning_results - QM9_D_EXP_physdays.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc

# %%
# compute data savings plot for QM9
os.chdir("/projappl/project_2000382/ghoshkun/code/Multi_Fidelity_Prediction_GP/mfgp/task1_new/puhti_runs")    
savings_aa, batch_size_aa = plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - AA_A_EXP_old.csv",
                  strategy_d_file = "csv_files/Active_learning_results - AA_D_EXP_old.csv",
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc
# [ 0.04438062 -0.2239867   0.03023715  0.52630918  0.93471799 -2.68499825]
batch_size_aa

# %%
plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - AA_A_2k.csv",
                  strategy_d_file = "csv_files/Active_learning_results - AA_D_2k.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc

# %%
plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - AA_A_1k.csv",
                  strategy_d_file = "csv_files/Active_learning_results - AA_D_1k.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc

# %%
plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - QM9_A_EXP_old.csv",
                  strategy_d_file = "csv_files/Active_learning_results - QM9_D_EXP_old.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc

# %%
savings_qm9, batch_size_qm9 = plot_data_savings(strategy_a_file = "csv_files/Active_learning_results - QM9_A_EXP.csv",
                  strategy_d_file = "csv_files/Active_learning_results - QM9_D_EXP.csv", 
                  batch_size=1,
                  percentage=True) # 1 for 1k, 2 for 2k etc


# %%
def plot_datasavings_qm9_aa():
    set_mpl_params(matplotlib)
    plt.plot(batch_size_qm9, savings_qm9, color="#1b9e77")
    plt.scatter(batch_size_qm9, savings_qm9, color="#1b9e77", label="QM9")

    plt.plot(batch_size_aa, savings_aa, color="k")
    plt.scatter(batch_size_aa, savings_aa, color="k", label="AA")

    # plt.xscale("linear")
    plt.legend()
    plt.xticks(batch_sizes)
    plt.xlabel("Dataset size in 1000s")
    plt.ylabel("Datasaving in percent")
    plt.grid()
    
plot_datasavings_qm9_aa()


# %%
def plot_figure_1(aa, qm9):
    aa_a = pd.read_csv(aa)
    qm9_a = pd.read_csv(qm9)

    set_mpl_params(matplotlib)
    plt.figure(figsize=(6,4), dpi=200)
    labs, labels = get_labels()
    
    batch_size = np.cumsum(qm9_a.batch_size // 1000)
    
    plt.plot(batch_size, qm9_a.mean_vals.to_numpy(), color="#1b9e77")
    plt.scatter(batch_size, qm9_a.mean_vals.to_numpy(), color="#1b9e77", label="QM9")
    
    plt.plot(batch_size, aa_a.mean_vals.to_numpy(), color="k")
    plt.scatter(batch_size, aa_a.mean_vals.to_numpy(), color="k", label="AA")
    
    plt.xticks(batch_size)
    # plt.xscale("linear")
    plt.legend()
    plt.xlabel("Training set size (in 1000s)")
    plt.ylabel("Mean Absolute Error (eV)")
    plt.grid()

plot_figure_1("csv_files/Active_learning_results - AA_A_EXP_old.csv", 
              "csv_files/Active_learning_results - QM9_A_EXP_old.csv")


# %%
def plot_abcd(aa_a, aa_b, aa_c, aa_d):
    
    a = pd.read_csv(aa_a)
    b = pd.read_csv(aa_b)
    c = pd.read_csv(aa_c)
    d = pd.read_csv(aa_d)

    set_mpl_params(matplotlib)
    plt.figure(figsize=(6,4), dpi=200)
    labs, labels = get_labels()
    
    batch_size = np.cumsum(a.batch_size // 1000)
    
    lines = {"linewidth": 6}
    matplotlib.rc('lines', **lines)
    
    plt.plot(batch_size, a.mean_vals.to_numpy(), color="tab:blue")
    plt.scatter(batch_size, a.mean_vals.to_numpy(), color="tab:blue", label="A")
    
    lines = {"linewidth": 4}
    matplotlib.rc('lines', **lines)
    
    plt.plot(batch_size, b.mean_vals.to_numpy(), color="tab:green")
    plt.scatter(batch_size, b.mean_vals.to_numpy(), color="tab:green", label="B")
     
    plt.plot(batch_size, c.mean_vals.to_numpy(), color="tab:red")
    plt.scatter(batch_size, c.mean_vals.to_numpy(), color="tab:red", label="C")
    
    plt.plot(batch_size, d.mean_vals.to_numpy(), color="tab:orange")
    plt.scatter(batch_size, d.mean_vals.to_numpy(), color="tab:orange", label="D")

    plt.xticks(batch_size)
    plt.xscale("linear")
    plt.legend()
    plt.xlabel("Training set size (in 1000s)")
    plt.ylabel("Mean Absolute Error (eV)")
    plt.grid()

plot_abcd("csv_files/Active_learning_results - AA_A_EXP_old.csv", 
          "csv_files/Active_learning_results - AA_B_EXP_old.csv",
          "csv_files/Active_learning_results - AA_C_EXP_old.csv",
          "csv_files/Active_learning_results - AA_D_EXP_old.csv")    

# %% [markdown]
# # Distribution of molecule count per energy range
#
# Done for AA and QM9

# %%
from matplotlib import pyplot as plt

import os
import numpy as np
import matplotlib
# %matplotlib inline

plt.style.use('seaborn-paper')
# font = {'size'   : 20}
# labelsize = {'labelsize'   : 20, 'titlesize' : 20}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **labelsize)
params = {'legend.fontsize': 'small',
  'axes.labelsize': 'medium',
  'axes.titlesize':'medium',
  'xtick.labelsize':'medium',
  'ytick.labelsize':'medium'}
matplotlib.rcParams.update(params)
set_mpl_params(matplotlib)

os.chdir("/projappl/project_2000382/ghoshkun/data")
aa = np.loadtxt("AA/HOMO.txt")
qm9 = np.loadtxt("QM9/HOMO.txt")


# %%
def plot_data_distribution(data, color):
    set_mpl_params(matplotlib)
    obj = plt.hist(data, bins=50, density=True, color=color,range=[-20, 0])
    plt.ylim((0,1))
    plt.xlabel("energy (eV)")
    plt.ylabel("Percent of dataset")
    plt.grid(True)
    
plot_data_distribution(aa, "#fecc5c")

# %%
# obj = plt.hist(qm9, bins=50, density=True, color="#fd8d3c",range=[-20, 0])  
# plt.ylim((0,1))
# plt.xlabel("energy (eV)")
# plt.ylabel("Percent of dataset")

plot_data_distribution(qm9, "#fd8d3c")


# %%
# plot figure 1
def plot_all_datasets():
    set_mpl_params(matplotlib)
    qm9 = pd.read_csv("csv_files/Active_learning_results - QM9_A_EXP_old.csv")
    aa  = pd.read_csv("csv_files/Active_learning_results - AA_A_EXP_old.csv")
    
    
