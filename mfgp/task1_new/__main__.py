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
from utils import desc_pp, desc_pp_notest, has_duplicates, pre_rem_split, r2_byhand
from io_utils import overwrite, append_write, out_condition, out_time, out_time_all, fig_atom, fig_HOMO
import datetime
import time
import multiprocessing as multi

def main():
    start_all = time.time()

    filepath = sys.argv[1]

    #-- input_data
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
    if save_load_split_flag == "True":
        save_load_split_flag = True
    elif save_load_split_flag == "False":
        save_load_split_flag = False
    else:
        append_write(out_name,"save_load_split_flag should be True or False \n")
        append_write(out_name,"program stopped ! \n")    

    #-- initialize
    overwrite(out_name, "")
    
    #-- path 
    mbtr_path = input_data_split[10]
    json_path = input_data_split[11]
    loadidxs_path = input_data_split[13]
    loadsplitidxs_path = input_data_split[15]

    #-- load
    start = time.time()
    append_write(out_name, "starting load mbtr \n")
    mbtr_data = load_npz(mbtr_path)
    process_time = time.time() - start
    out_time(out_name, process_time)

    start = time.time()
    append_write(out_name, "starting load df_62k \n")
    df_62k = pd.read_json(json_path, orient='split')
    process_time = time.time() - start
    out_time(out_name, process_time)

    num_atoms = df_62k["number_of_atoms"].values

    homo_lowfid = df_62k.apply(lambda row: get_level(
        row, level_type='HOMO', subset='PBE+vdW_vacuum'),
                               axis=1).to_numpy()

    #-- output
    f = open(out_name,"a")
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
    f.write("save_load_split_flag " + str(save_load_split_flag) + "\n" )
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
    f.close()

    #-- figure of 62k dataset
    fig_atom(df_62k,range(61489),"atoms_all.eps")
    fig_HOMO(homo_lowfid,range(61489), "HOMO_all.eps")

    #-- Setup the GP model
    n_features = int(mbtr_data_red.shape[1])
    kernel = RBF(length_scale=[1e7], length_scale_bounds=(1e7, 1e8))
    gpr = GaussianProcessRegressor(kernel=kernel)                                                                                             

    if save_load_flag == "save" :
        #-- load prepared dataset
        if (save_load_split_flag):
            append_write(out_name, "save_load_split_flag: True \n")
            load_idxs = np.load(loadsplitidxs_path)
            test_idxs = load_idxs['test_idxs']
            remaining_idxs = load_idxs['remaining_idxs']
            append_write(out_name, "1st index of test_idxs " + str(test_idxs[0]) + "\n" )
            append_write(out_name, "length of loaded split test_idxs " + str(len(test_idxs)) + "\n")
            append_write(out_name, "length of loaded split remaining_idxs " + str(len(remaining_idxs)) + "\n")
        #-- not load prepared dataset, use original 62k dataset
        else:
            append_write(out_name, "save_load_split_flag: False \n")
            #-- mbtr data size
            if dataset_size < mbtr_data_size:
                append_write(out_name, "mbtr_data_size =/ dataset_size \n")
                append_write(out_name, "mbtr_data_size " + str(mbtr_data_size) + "\n")
                append_write(out_name, "dataset_size " + str(dataset_size) + "\n")

                remaining_idxs, not_used_idxs = train_test_split(range(mbtr_data_size), train_size = dataset_size)
                np.save(out_name + "_dataset_idxs", remaining_idxs)

                #-- figure
                fig_atom(df_62k,remaining_idxs, out_name +'_atoms_pre_dataset.eps')
                fig_HOMO(homo_lowfid,remaining_idxs, out_name + 'HOMO_pre_dataset.eps')

                #-- test , 1st train, remaining split.
                test_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = test_set_size)
                np.savez(out_name + "_dataset_split_idxs.npz", remaining_idxs=remaining_idxs, test_idxs=test_idxs)

            elif dataset_size  == mbtr_data_size:    
                append_write(out_name, "mbtr_data_size ==  dataset_size \n")
                test_idxs, remaining_idxs = train_test_split(range(mbtr_data_size), train_size = test_set_size)

            else: 
                append_write(out_name,"dataset_size should be = or < mbtr_data_size \n")
                append_write(out_name,"program stopped ! \n")
                sys.exit()

            append_write(out_name, "length of test_idxs " + str(len(test_idxs)) + "\n")
            append_write(out_name, "length of remaining_idxs " + str(len(remaining_idxs)) + "\n")            

        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs)
        append_write(out_name, "length of prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name, "length of remaining_idxs " + str(len(remaining_idxs)) + "\n")

        #-- save the values
        append_write(out_name, "save_load_flag: save \n")
        np.savez(out_name + "_0_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, test_idxs=test_idxs)

    #-- load index
    elif save_load_flag == "load":
        append_write(out_name, "save_load_flag: load \n")
        load_idxs = np.load(loadidxs_path)
        remaining_idxs = load_idxs['remaining_idxs']
        prediction_idxs = load_idxs['prediction_idxs']
        test_idxs = load_idxs['test_idxs']
        append_write(out_name, "1st index of test_idxs " + str(test_idxs[0]) + "\n")
        append_write(out_name, "length of loaded test_idxs " + str(len(test_idxs)) + "\n")
        append_write(out_name, "length of loaded prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name, "length of loaded remaining_idxs " + str(len(remaining_idxs)) + "\n")
        
    #-- save_load_flag should be "save" or "load"
    else:
        append_write(out_name,"save_load_flag should be save or load \n")
        append_write(out_name,"program stopped ! \n")
        sys.exit()
        
    #-- 
    X_train, X_test = mbtr_data_red[prediction_idxs, :], mbtr_data_red[test_idxs, :]
    y_train, y_test = homo_lowfid[prediction_idxs], homo_lowfid[test_idxs]

    #-- Preprocessing
    X_train_pp, X_test_pp = desc_pp(preprocess, X_train, X_test)

    #-- Fit the model
    append_write(out_name,"initial learning \n")
    append_write(out_name,"start to make a model \n")
    start = time.time()
    gpr.fit(X_train_pp, y_train)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- Predict
    start = time.time()
    append_write(out_name,"start prediction" + "\n")
    mu_s, std_s = gpr.predict(X_test_pp, return_std=True)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- score(r^2 by hand)
    start = time.time()
    append_write(out_name,"start calculating r2 by hand \n")
    r2 = r2_byhand(y_test, mu_s)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- score(MAE)
    start = time.time()
    append_write(out_name,"start calculating MAE \n")
    MAE = np.array(mean_absolute_error(y_test, mu_s))
    process_time = time.time() - start
    out_time(out_name, process_time)

    append_write(out_name,"r2 by hand " + str(r2) + "\n")
    append_write(out_name,"MAE " + str(MAE) + "\n")

    #-- score array
    sample_sum = np.array(len(prediction_idxs))
    r2_sum = r2
    MAE_sum = MAE

    #-- save score
    np.save(out_name + "_sample", sample_sum)
    np.save(out_name + "_r2", r2_sum)    
    np.save(out_name + "_MAE", MAE_sum)
    np.savez(out_name + "_score.npz", sample=sample_sum, r2=r2_sum,MAE=MAE_sum)

    #-- save mean and std
    np.save(out_name + "_mu_0",mu_s)
    np.save(out_name + "_std_0",std_s)

    #-- figure
    fig_atom(df_62k, prediction_idxs, out_name +'_atoms_pre_0.eps')
    fig_HOMO(homo_lowfid,prediction_idxs, out_name + 'HOMO_pre_0.eps')

    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)

    for i in range(num_itr):
        append_write(out_name,"============================= \n")
        append_write(out_name,str(i+1) + "-th learning" + "\n")

        prediction_idxs, remaining_idxs, X_train_pp, y_train = acq_fn(fn_name,i,prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, mbtr_data_red, homo_lowfid, K_high, gpr, preprocess, out_name)
        append_write(out_name,"check duplicates of prediction_idxs " + str(has_duplicates(prediction_idxs)) + "\n")
        all_idxs = np.concatenate([test_idxs, prediction_idxs, remaining_idxs], 0)
        append_write(out_name,"length if all_idxs " + str(len(all_idxs)) + "\n")
        append_write(out_name,"check duplicates of all idxs " + str(has_duplicates(all_idxs)) + "\n")
        append_write(out_name,"length of prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name,"length of remaining_idxs " + str(len(remaining_idxs)) + "\n")

        #-- Fit the model
        start = time.time()
        append_write(out_name,"start to make a model \n")
        gpr.fit(X_train_pp, y_train)
        process_time = time.time() - start
        out_time(out_name, process_time)    

        #-- predict
        start = time.time()
        append_write(out_name,"start prediction \n")
        mu_s, std_s = gpr.predict(X_test_pp, return_std=True) #mu->mean? yes
        process_time = time.time() - start
        out_time(out_name, process_time)

        #-- score(r^2 by hand)
        start = time.time()
        append_write(out_name,"start calculating r2 by hand \n")
        r2 = r2_byhand(y_test, mu_s)
        process_time = time.time() - start
        out_time(out_name, process_time)

        #-- score(MAE)
        append_write(out_name,"start calculating MAE \n")
        start = time.time()
        MAE = np.array(mean_absolute_error(y_test, mu_s))
        process_time = time.time() - start
        out_time(out_name, process_time)

        append_write(out_name,"r2 by hand " + str(r2) + "\n")
        append_write(out_name,"MAE " + str(MAE) + "\n")

        #--
        sample_sum = np.append(sample_sum ,len(prediction_idxs))
        r2_sum = np.append(r2_sum ,r2)
        MAE_sum = np.append(MAE_sum, MAE)

        #-- save 
        np.save(out_name + "_sample", sample_sum)
        np.save(out_name + "_r2", r2_sum)    
        np.save(out_name + "_MAE", MAE_sum)
        np.savez(out_name + "_score.npz", sample=sample_sum, r2=r2_sum,MAE=MAE_sum)

        #-- save mean and std
        np.save(out_name + "_mu_" + str(i+1),mu_s)
        np.save(out_name + "_std_" + str(i+1),std_s)

        #-- figure
        fig_atom(df_62k, prediction_idxs, out_name + '_atoms_pre_' + str(i+1) +'.eps')
        fig_HOMO(homo_lowfid,prediction_idxs, out_name + '_HOMO_pre_' + str(i+1) + '.eps')

        process_time_all = time.time() - start_all
        out_time_all(out_name, process_time_all)

    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)

    #-- score summary
    f = open(out_name, 'a')
    form="%i %20.10f %20.10f \n"
    for i in range(num_itr + 1):
        f.write(form % (i,r2_sum[i],MAE_sum[i]))
    f.close()
    

#-- Start active_learning
if __name__ == "__main__":
        main()
