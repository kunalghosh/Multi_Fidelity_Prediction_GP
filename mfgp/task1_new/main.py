import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from mfgp.utils import get_level
from acq import acq_fn
from utils import desc_pp, desc_pp_notest, has_duplicates, pre_rem_split
from io_utils import overwrite, append_write, out_condition
import datetime
import time
#--
import multiprocessing as multi

#text1 ="test1 \n"
#text2 ="test2"

#overwrite("./test.dat", text1)
#append_write("./test.dat", text2)

start_all = time.time()

filepath = sys.argv[1]

f = open(filepath)
input_data = f.read()
f.close()
input_data_split = input_data.split('\n')

#-- condition
fn_name = input_data_split[0] #acqusition
out_name = input_data_split[1] #output
num_itr = int(input_data_split[2])
K_high = int(input_data_split[3])
dataset_size = int(input_data_split[4])  #61489
test_set_size = int(input_data_split[5])
prediction_set_size = int(input_data_split[6]) # usually training set is much larger so 500 is reasonable 
rnd_size = float(input_data_split[7])
mbtr_red_tol = int(input_data_split[8])
mbtr_red = input_data_split[12]
preprocess = input_data_split[9]
save_load_flag = input_data_split[14]
save_load_split_flag = input_data_split[16]

#-- path 
mbtr_path = input_data_split[10]
json_path = input_data_split[11]
loadidxs_path = input_data_split[13]
loadsplitidxs_path = input_data_split[15]

#-- load
f = open(out_name,"w")
start = time.time()
f.write("starting load mbtr" + "\n"),f.flush()
mbtr_data = load_npz(mbtr_path)
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

start = time.time()
f.write("starting load df_62k" + "\n"),f.flush()
df_62k = pd.read_json(json_path, orient='split')
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

homo_lowfid = df_62k.apply(lambda row: get_level(
    row, level_type='HOMO', subset='PBE+vdW_vacuum'),
                           axis=1).to_numpy()


#-- output
f.write("=============================" + "\n")
dt_now = datetime.datetime.now()
f.write(str(dt_now) + "\n" )
f.write("condition" + "\n" )
f.write("fn_name(acquistion function) " + fn_name + "\n" )
f.write("num_itr " + str(num_itr) + "\n" )
f.write("dataset_size " + str(dataset_size) + "\n" )
f.write("test_set_size " + str(test_set_size) + "\n" )
f.write("prediction_set_size " + str(prediction_set_size) + "\n" )
f.write("rnd_size(random sampling) " + str(rnd_size) + "\n" )
f.write("high_size(high std) " + str(K_high) + "\n" )
f.write("tol for reducing the MBTR array " + str(mbtr_red_tol) + "\n" )
f.write("flag for reducing the MBTR array " + str(mbtr_red) + "\n" )
f.write("mbtr_path " + mbtr_path + "\n" )
f.write("json_path " + json_path + "\n" )
f.write("mbtr_data col " + str(mbtr_data.shape[0]) + "\n" )
f.write("mbtr_data row " + str(mbtr_data.shape[1]) + "\n" )
mbtr_data_size = mbtr_data.shape[0]
f.write("mbtr_data_size " + str(mbtr_data_size) + "\n" )
f.write("save_load_flag " + save_load_flag + "\n" )
f.write("loadidxs_path " + loadidxs_path + "\n" )
f.write("save_load_split_flag " + save_load_split_flag + "\n" )
f.write("loadsplitidxs_path " + loadsplitidxs_path + "\n" )

f.write("CPU: " + str(multi.cpu_count()) + "\n" )

#-- reduce the size of descriptor array
if mbtr_red == "yes":
    mbtr_data_red = mbtr_data[:,mbtr_data.getnnz(0) > mbtr_red_tol]
else:
    mbtr_data_red = mbtr_data

f.write("mbtr_data_red_size " + str(mbtr_data_red.shape[1]) + "\n" )
f.write("preprocess " + preprocess + "\n" )
f.write("=============================" + "\n")
f.flush()

#-- figure
plt.figure()
num_atoms = df_62k["number_of_atoms"].values
plt.title('', fontsize = 20)
plt.xlabel('Number of atoms', fontsize = 16)
plt.ylabel('Number of molecules', fontsize = 16)
plt.tick_params(labelsize=14)
plt.xlim(0, 90)
plt.ylim(0, 2500)
(a_hist2, a_bins2, _) = plt.hist(num_atoms[:], bins=170)

plt.savefig('atoms_all.eps') 

plt.figure()
plt.title('Histogram of HOMO energy', fontsize = 20)
plt.xlabel('Energy', fontsize = 16)
plt.ylabel('', fontsize = 16)
plt.tick_params(labelsize=14)
plt.xlim(-10, 0)
plt.ylim(0, 6000)
(a_hist2, a_bins2, _) = plt.hist(homo_lowfid[:], bins=70)
plt.savefig('HOMO_all.eps') 

#-- Setup the GP model
n_features = int(mbtr_data_red.shape[1])
kernel = RBF(length_scale=[1e7], length_scale_bounds=(1e7, 1e8))
gpr = GaussianProcessRegressor(kernel=kernel)                                                                                             

if save_load_flag == "save" :
    if save_load_split_flag !="load":
        #-- mbtr data size
        if dataset_size != mbtr_data_size:
            f.write("mbtr_data_size =/" + " dataset_size" + "\n"),f.flush()
            f.write("mbtr_data_size " + str(mbtr_data_size) + "\n"),f.flush()
            f.write("dataset_size " + str(dataset_size) + "\n"),f.flush()
            remaining_idxs, not_used_idxs = train_test_split(range(mbtr_data_size), train_size = dataset_size)
            np.save(out_name + "_dataset_idxs", remaining_idxs)

            #-- figure
            plt.figure()
            plt.title('', fontsize = 20)
            plt.xlabel('Number of atoms', fontsize = 16)
            plt.ylabel('Number of molecules', fontsize = 16)
            plt.tick_params(labelsize=14)
            plt.xlim(0, 90)
            plt.ylim(0, 2500)
            (a_hist2, a_bins2, _) = plt.hist(num_atoms[remaining_idxs], bins=170)
            plt.savefig( out_name +'_atoms_pre_dataset.eps') 
        
            plt.figure()
            plt.title('Histogram of HOMO energy', fontsize = 20)
            plt.xlabel('Energy', fontsize = 16)
            plt.ylabel('', fontsize = 16)
            plt.tick_params(labelsize=14)
            plt.xlim(-10, 0)
            plt.ylim(0, 6000)
            (a_hist2, a_bins2, _) = plt.hist(homo_lowfid[remaining_idxs], bins=70)
            plt.savefig( out_name + 'HOMO_pre_dataset.eps') 
        
            #-- test , 1st train, remaining split.
            test_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = test_set_size)#, random_state=0)
            np.savez(out_name + "_dataset_split_idxs.npz", remaining_idxs=remaining_idxs, test_idxs=test_idxs)
            
        elif dataset_size  == mbtr_data_size:    
            f.write("mbtr_data_size ==" + " dataset_size" + "\n"),f.flush()
            test_idxs, remaining_idxs = train_test_split(range(mbtr_data_size), train_size = test_set_size)#, random_state=0)
            
            f.write("1st index of test_idxs " + str(test_idxs[0]) + "\n" ),f.flush()
            f.write("length of test_idxs " + str(len(test_idxs)) + "\n"),f.flush()
            f.write("length of remaining_idxs " + str(len(remaining_idxs)) + "\n"),f.flush()
                    
    #-- load split index
    elif save_load_split_flag =="load" :
        f.write("save_load_split_flag: " + "load" + "\n"),f.flush()
        load_idxs = np.load(loadsplitidxs_path)
        test_idxs = load_idxs['test_idxs']
        remaining_idxs = load_idxs['remaining_idxs']
        f.write("1st index of test_idxs " + str(test_idxs[0]) + "\n" ),f.flush()
        f.write("length of loaded split test_idxs " + str(len(test_idxs)) + "\n"),f.flush()
        f.write("length of loaded split remaining_idxs " + str(len(remaining_idxs)) + "\n"),f.flush()

    prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)
    f.write("length of prediction_idxs " + str(len(prediction_idxs)) + "\n"),f.flush()
    f.write("length of remaining_idxs " + str(len(remaining_idxs)) + "\n"),f.flush()

    #-- save the values
    f.write("save_load_flag: " + "save" + "\n"),f.flush()
    np.savez(out_name + "_0_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, test_idxs=test_idxs)

#-- load index
elif save_load_flag == "load":
    f.write("save_load_flag: " + "load" + "\n"),f.flush()
    load_idxs = np.load(loadidxs_path)
    remaining_idxs = load_idxs['remaining_idxs']
    prediction_idxs = load_idxs['prediction_idxs']
    test_idxs = load_idxs['test_idxs']
    f.write("1st index of test_idxs " + str(test_idxs[0]) + "\n" ),f.flush()
    f.write("length of loaded test_idxs " + str(len(test_idxs)) + "\n"),f.flush()
    f.write("length of loaded prediction_idxs " + str(len(prediction_idxs)) + "\n"),f.flush()
    f.write("length of loaded remaining_idxs " + str(len(remaining_idxs)) + "\n"),f.flush()
    
X_train, X_test = mbtr_data_red[prediction_idxs, :], mbtr_data_red[test_idxs, :]
y_train, y_test = homo_lowfid[prediction_idxs], homo_lowfid[test_idxs]

#-- normalization
start = time.time()
f.write("starting preprocess " + "\n"),f.flush()
X_train_pp, X_test_pp = desc_pp(preprocess, X_train, X_test)
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- Fit the model
f.write("initial learning" + "\n"),f.flush()
start = time.time()
f.write("start to make a model" + "\n"),f.flush()
gpr.fit(X_train_pp, y_train)
#gpr.fit(X_train, y_train)
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- predict
start = time.time()
f.write("start prediction" + "\n"),f.flush()
mu_s, std_s = gpr.predict(X_test_pp, return_std=True)
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- score(r^2)
#score = np.array(gpr.score(X_test, y_test))
#start = time.time()
#f.write("start calculating score" + "\n"),f.flush()
#score = np.array(gpr.score(X_test_pp, y_test))  
#process_time = time.time() - start
#f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- score(r^2 by hand)
start = time.time()
f.write("start calculating r2 by hand " + "\n"),f.flush()
y_true = y_test
y_pred = mu_s
u = ((y_true - y_pred)**2).sum()
v = ((y_true - y_true.mean()) ** 2).sum()
r2 = 1-u/v
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- score(MAE)
start = time.time()
f.write("start calculating MAE" + "\n"),f.flush()
MAE = np.array(mean_absolute_error(y_test, mu_s))
process_time = time.time() - start
f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#f.write("score " + str(score) + "\n")
f.write("r2 by hand " + str(r2) + "\n")
f.write("MAE " + str(MAE) + "\n")

#-- score array
sample_sum = np.array(len(prediction_idxs))
#score_sum = score
r2_sum = r2
MAE_sum = MAE

#-- save score
#np.save("sample_"+ out_name, sample_sum)
#np.save("score_"+ out_name, score_sum)
#np.save("score2_"+ out_name, score2_sum)    
#np.save("MAE_"+ out_name, MAE_sum)
np.save(out_name + "_sample", sample_sum)
#np.save(out_name + "_score", score_sum)
np.save(out_name + "_r2", r2_sum)    
np.save(out_name + "_MAE", MAE_sum)
np.savez(out_name + "_score.npz", sample=sample_sum, r2=r2_sum,MAE=MAE_sum)

#-- save mean and std
np.save(out_name + "_mu_0",mu_s)
np.save(out_name + "_std_0",std_s)

#-- figure
plt.figure()
plt.title('', fontsize = 20)
plt.xlabel('Number of atoms', fontsize = 16)
plt.ylabel('Number of molecules', fontsize = 16)
plt.tick_params(labelsize=14)
plt.xlim(0, 90)
plt.ylim(0, 2500)
(a_hist2, a_bins2, _) = plt.hist(num_atoms[prediction_idxs], bins=170)
plt.savefig( out_name +'_atoms_pre_0.eps') 

plt.figure()
plt.title('Histogram of HOMO energy', fontsize = 20)
plt.xlabel('Energy', fontsize = 16)
plt.ylabel('', fontsize = 16)
plt.tick_params(labelsize=14)
plt.xlim(-10, 0)
plt.ylim(0, 6000)
(a_hist2, a_bins2, _) = plt.hist(homo_lowfid[prediction_idxs], bins=70)
plt.savefig( out_name + 'HOMO_pre_0.eps') 

#-- KRR for check
#from sklearn.kernel_ridge import KernelRidge
#start = time.time()
#f.write("start calculating krr" + "\n"),f.flush()
#krr = KernelRidge(kernel='rbf',alpha=1e-5,gamma=1e-8)
#krr.fit(X_train_pp, y_train)
#y_krr = krr.predict(X_test_pp)
#krr_score = krr.score(X_test_pp, y_test)
#f.write("krr_score " + str(krr_score) + "\n")
#krr_mae = mean_absolute_error(y_test, y_krr)
#f.write("krr_mae " + str(krr_mae) + "\n")    
#process_time = time.time() - start
#f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#-- KRR for check2
#noise = 0.4
#kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
#start = time.time()
#f.write("start calculating krr_2" + "\n"),f.flush()
#krr = KernelRidge(kernel=kernel)#,alpha=noise**2)
#krr.fit(X_train_pp, y_train)
#y_krr = krr.predict(X_test_pp)
#krr_score = krr.score(X_test_pp, y_test)
#f.write("krr_score_2 " + str(krr_score) + "\n")
#krr_mae = mean_absolute_error(y_test, y_krr)
#f.write("krr_mae_2 " + str(krr_mae) + "\n")    
#process_time = time.time() - start
#f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

process_time_all = time.time() - start_all
f.write("time all "),f.write(str(process_time_all) + "[s]" + "\n"),f.flush()

for i in range(num_itr):
    f.write("=============================" + "\n")
    f.write(str(i+1) + "-th learning" + "\n"),f.flush()

    prediction_idxs, remaining_idxs, X_train_pp, y_train = acq_fn(fn_name,i,prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, mbtr_data_red, homo_lowfid, K_high, gpr, preprocess, out_name)
    f.write("check duplicates of prediction_idxs " + str(has_duplicates(prediction_idxs)) + "\n"),f.flush()
    all_idxs = np.concatenate([test_idxs, prediction_idxs, remaining_idxs], 0)
    f.write("length if all_idxs " + str(len(all_idxs)) + "\n"),f.flush()
    f.write("check duplicates of all idxs " + str(has_duplicates(all_idxs)) + "\n"),f.flush()
    
    f.write("length of prediction_idxs " + str(len(prediction_idxs)) + "\n"),f.flush()
    f.write("length of remaining_idxs " + str(len(remaining_idxs)) + "\n"),f.flush()

    #-- Fit the model
    start = time.time()
    f.write("start to make a model" + "\n")
    gpr.fit(X_train_pp, y_train)
    process_time = time.time() - start
    f.write("time "),f.write(str(process_time) + "[s]" + "\n")
    
    #-- predict
    start = time.time()
    f.write("start prediction" + "\n"),f.flush()
    mu_s, std_s = gpr.predict(X_test_pp, return_std=True) #mu->mean? yes
    process_time = time.time() - start
    f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#    f.write("start calculating score" + "\n"),f.flush()
#    start = time.time()
#    score = gpr.score(X_test_pp, y_test)
#    process_time = time.time() - start
#    f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

    #---------
    start = time.time()
    f.write("start calculating r2 by hand " + "\n"),f.flush()
    y_true = y_test
    y_pred = mu_s
    u = ((y_true - y_pred)**2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1-u/v
    process_time = time.time() - start
    f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()
    #---------

    f.write("start calculating MAE" + "\n"),f.flush()
    start = time.time()
    MAE = np.array(mean_absolute_error(y_test, mu_s))
    process_time = time.time() - start
    f.write("time "),f.write(str(process_time) + "[s]" + "\n"),f.flush()

#    f.write("score " + str(score) + "\n")
    f.write("r2 by hand " + str(r2) + "\n")    
    f.write("MAE " + str(MAE) + "\n")
#    print("score",i,gpr.score(X_test_pp, y_test))

    #--
    sample_sum = np.append(sample_sum ,len(prediction_idxs))
#    score_sum = np.append(score_sum ,score)
    r2_sum = np.append(r2_sum ,r2)
    MAE_sum = np.append(MAE_sum, MAE)

    #-- save 
    np.save(out_name + "_sample", sample_sum)
#    np.save(out_name + "_score", score_sum)
    np.save(out_name + "_r2", r2_sum)    
    np.save(out_name + "_MAE", MAE_sum)
    np.savez(out_name + "_score.npz", sample=sample_sum, r2=r2_sum,MAE=MAE_sum)

    #-- save mean and std
    np.save(out_name + "_mu_" + str(i+1),mu_s)
    np.save(out_name + "_std_" + str(i+1),std_s)
    
    #-- figure
    plt.figure()
    plt.title('', fontsize = 20)
    plt.xlabel('Number of atoms', fontsize = 16)
    plt.ylabel('Number of molecules', fontsize = 16)
    plt.tick_params(labelsize=14)
    plt.xlim(0, 90)
    plt.ylim(0, 2500)
    (a_hist2, a_bins2, _) = plt.hist(num_atoms[prediction_idxs], bins=170)
    plt.savefig( out_name + '_atoms_pre_' + str(i+1) +'.eps') 

    plt.figure()
    plt.title('Histogram of HOMO energy', fontsize = 20)
    plt.xlabel('Energy', fontsize = 16)
    plt.ylabel('', fontsize = 16)
    plt.tick_params(labelsize=14)
    plt.xlim(-10, 0)
    plt.ylim(0, 6000)
    (a_hist2, a_bins2, _) = plt.hist(homo_lowfid[prediction_idxs], bins=70)
    plt.savefig( out_name + 'HOMO_pre_' + str(i+1) + '.eps') 
    
    process_time_all = time.time() - start_all
    f.write("time all "),f.write(str(process_time_all) + "[s]" + "\n"),f.flush()
    
process_time_all = time.time() - start_all
f.write("time all "),f.write(str(process_time_all) + "[s]" + "\n"),f.flush()

f.write("R2 "),f.write(str(r2_sum) + "\n"),f.flush()
f.write("MAE "),f.write(str(MAE_sum)),f.flush()
f.close()
