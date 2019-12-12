import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz, save_npz, lil_matrix
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from mfgp.utils import get_level
from acq import acq_fn
from utils import desc_pp, desc_pp_notest, has_duplicates, pre_rem_split, r2_byhand
from io_utils import overwrite, append_write, out_condition, out_time, out_time_all, out_time_all_temp, fig_atom, fig_HOMO, fig_scatter_r2, fig_MDS_scatter_std, fig_MDS_scatter_label
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
    dataset = input_data_split[17]
    length = float(input_data_split[18])
    const = float(input_data_split[19])
    bound = float(input_data_split[20])
    n_opt = int(input_data_split[21])
    kernel_type = input_data_split[22]
    restart_flag = input_data_split[23] 
    
    pre_idxs = np.empty(int(num_itr+1),dtype = int)
    
    for i in range(int(num_itr+1)):
        pre_idxs_temp = i + 24
        pre_idxs[i] = input_data_split[pre_idxs_temp]

#    if save_load_split_flag == "True":
#        save_load_split_flag = True
#    elif save_load_split_flag == "False":
#        save_load_split_flag = False
#    else:
#        append_write(out_name,"save_load_split_flag should be True or False \n")
#        append_write(out_name,"program stopped ! \n")    
#        sys.exit()

    #--
    save_load_split_flag = str_to_bool(save_load_split_flag)
    restart_flag = str_to_bool(restart_flag)    

    #-- initialize
    if restart_flag:
        append_write(out_name, "\n\n restart \n\n")
    else:
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

    if dataset == "OE" :
        start = time.time()
        append_write(out_name, "starting load df_62k \n")
        df_62k = pd.read_json(json_path, orient='split')
        process_time = time.time() - start
        out_time(out_name, process_time)
        
        num_atoms = df_62k["number_of_atoms"].values
        
        homo_lowfid = df_62k.apply(lambda row: get_level(
            row, level_type='HOMO', subset='PBE+vdW_vacuum'),
                                   axis=1).to_numpy()
    elif dataset == "AA":
        start = time.time()
        append_write(out_name, "starting load AA \n")
        homo_lowfid =  np.loadtxt(json_path)
        process_time = time.time() - start
        out_time(out_name, process_time)
    else:
        append_write(out_name,"Dataset sould be AA or OE \n")
        append_write(out_name,"program stopped ! \n")    
        sys.exit()
        
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
    f.write("rnd_size(random sampling) or K_pre(high_and_cluster) " + str(rnd_size) + "\n" )
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
    f.write("dataset is " + str(dataset) + "\n" )
    for i in range(int(num_itr + 1)):
        f.write( " the number of added indexes in step " +str(i) + " " + str(pre_idxs[i]) + "\n" )
    
    #-- reduce the size of descriptor array
    if mbtr_red == "yes":
        mbtr_data_red = mbtr_data[:,mbtr_data.getnnz(0) > mbtr_red_tol]
    else:
        mbtr_data_red = mbtr_data
        
    f.write("mbtr_data_red_size " + str(mbtr_data_red.shape[1]) + "\n" )
    f.write("preprocess " + preprocess + "\n" )
    f.write("length_value " + str(length) + "\n" )
    f.write("const_value " + str(const) + "\n" )
    f.write("upper bound " + str(bound) + "\n" )
    f.write("lower bound " + str(1.0/bound) + "\n" )
    f.write("n_restart_optimizer " + str(n_opt) + "\n" )        
    f.write("=============================" + "\n")
    f.flush()
    f.close()

    #-- figure of 62k dataset and AA dataset
    if dataset == "OE" :
        fig_atom(df_62k,range(61489),"atoms_all.eps")
        fig_HOMO(homo_lowfid,range(61489), "HOMO_all.eps")
    elif dataset == "AA" :
        fig_HOMO(homo_lowfid,range(44004), "HOMO_all.eps")
    else:
        append_write(out_name,"Dataset sould be AA or OE \n")
        append_write(out_name,"program stopped ! \n")    
        sys.exit()
        
    #-- Setup the GP model
    n_features = int(mbtr_data_red.shape[1])
    if kernel_type == "RBF":
        kernel = RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound))
    elif kernel_type == "constRBF":
        kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) # best result.    
    else:
        append_write(out_name,"kernel sould be RBF or constRBF \n")
        append_write(out_name,"program stopped ! \n")    
        sys.exit()
        
    normalize_y = False

    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer = n_opt)

    if save_load_flag == "save" :
        #-- load prepared dataset
        if save_load_split_flag:
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
                if dataset == "OE" :
                    fig_atom(df_62k,remaining_idxs, out_name +'_atoms_pre_dataset.eps')
                    fig_HOMO(homo_lowfid,remaining_idxs, out_name + 'HOMO_pre_dataset.eps')
                elif dataset == "AA" :
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

        prediction_set_size = pre_idxs[0]
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
    para_kernel_aft = gpr.kernel_.get_params()
    process_time = time.time() - start
    out_time(out_name, process_time)

    with open(out_name + '_para_kernel_aft_0.txt', 'w') as f2:
        print(para_kernel_aft, file=f2)

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

    #-- plot r2
    fig_scatter_r2(y_test, mu_s, out_name + "_r2_0.eps")
    #    plt.figure()
#    plt.title('', fontsize = 20)
#    plt.xlabel('reference', fontsize = 16)
#    plt.ylabel('predicted', fontsize = 16)
#    plt.tick_params(labelsize=14)
#    plt.scatter(y_test, mu_s)
#    plt.savefig(out_name + "_r2_0.eps")
    
    #-- Fit the model (Fixed)
    #    kernel = ConstantKernel(constant_value_bounds=(1e-5, 1e5)) * RBF(length_scale=[7071], length_scale_bounds=(1e-5, 1e5)) # best result.
#    kernel = RBF(length_scale=[7071], length_scale_bounds=(1e2, 1e6))
#    gpr = GaussianProcessRegressor(kernel=kernel, optimizer = None,normalize_y=normalize_y, n_restarts_optimizer=2)
#    append_write(out_name,"initial learning fixed \n")
#    append_write(out_name,"start to make a model \n")
#    start = time.time()
#    para_bef = gpr.get_params()
#    gpr.fit(X_train_pp, y_train)
#    para_aft = gpr.get_params()
#    para_kernel_aft = gpr.kernel_.get_params()
#    process_time = time.time() - start
#    out_time(out_name, process_time)##

#    with open('para_bef_0_fixed.txt', 'w') as f1:
#        print(para_bef, file=f1)
#    with open('para_aft_0_fixed.txt', 'w') as f2:
#        print(para_aft, file=f2)
#        #
#    with open('para_kernel_aft_0_fixed.txt', 'w') as f2:
#        print(para_kernel_aft, file=f2)

    #-- Predict
#    start = time.time()
#    append_write(out_name,"start prediction \n")
#    mu_s, std_s = gpr.predict(X_test_pp, return_std=True)
#    process_time = time.time() - start

    #-- score(r^2 by hand)
#    start = time.time()
#    append_write(out_name,"start calculating r2 by hand \n")
#    r2 = r2_byhand(y_test, mu_s)
#    process_time = time.time() - start
#    out_time(out_name, process_time)

    #-- score(MAE)
#    start = time.time()
#    append_write(out_name,"start calculating MAE \n")
#    MAE = np.array(mean_absolute_error(y_test, mu_s))
#    process_time = time.time() - start
#    out_time(out_name, process_time)
#    append_write(out_name,"r2 by hand fixed " + str(r2) + "\n")
#    append_write(out_name,"MAE fixed " + str(MAE) + "\n")

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
    if dataset == "OE" :
        fig_atom(df_62k, prediction_idxs, out_name +'_atoms_pre_0.eps')
        fig_HOMO(homo_lowfid, prediction_idxs, out_name + 'HOMO_pre_0.eps')
    elif dataset == "AA" :
        fig_HOMO(homo_lowfid, prediction_idxs, out_name + 'HOMO_pre_0.eps')
        
    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)

    
    #-- iteration start
    for i in range(num_itr):
        append_write(out_name,"============================= \n")
        append_write(out_name,str(i+1) + "-th learning" + "\n")
        start_all_temp = time.time()

        prediction_set_size = pre_idxs[i+1]

        prediction_idxs, remaining_idxs, X_train_pp, y_train = acq_fn(fn_name,i,prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, mbtr_data_red, homo_lowfid, K_high, gpr, preprocess, out_name)

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_full_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs = prediction_idxs, test_idxs = test_idxs)

        append_write(out_name,"check duplicates of prediction_idxs " + str(has_duplicates(prediction_idxs)) + "\n")
        all_idxs = np.concatenate([test_idxs, prediction_idxs, remaining_idxs], 0)
        append_write(out_name,"length of all_idxs " + str(len(all_idxs)) + "\n")
        append_write(out_name,"check duplicates of all idxs " + str(has_duplicates(all_idxs)) + "\n")
        append_write(out_name,"length of prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name,"length of remaining_idxs " + str(len(remaining_idxs)) + "\n")
        append_write(out_name,"length of test_idxs " + str(len(test_idxs)) + "\n")

        #-- get kernel parameter
        if kernel_type == "RBF":
            length = para_kernel_aft['length_scale']
            kernel = RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound))
        elif kernel_type == "constRBF":
            const = para_kernel_aft['k1__constant_value']
            length = para_kernel_aft['k2__length_scale']
            kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) # best result.    

        gpr = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt)            
        append_write(out_name,"length of RBF kernel " + str(length) + "\n")
        append_write(out_name,"constant of constant kernel " + str(const) + "\n")
        
        #-- Fit the model
        start = time.time()
        append_write(out_name,"start to make a model \n")
        gpr.fit(X_train_pp, y_train)
        process_time = time.time() - start
        out_time(out_name, process_time)    

        para_kernel_aft = gpr.kernel_.get_params()
        with open(out_name + '_para_kernel_aft_' + str(i+1) +'.txt', 'w') as f2:
            print(para_kernel_aft, file=f2)
    
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

        #-- plot r2
        fig_scatter_r2(y_test, mu_s, out_name + "_r2_" + str(i+1) + ".eps")
#        plt.figure()
#        plt.title('', fontsize = 20)
#        plt.xlabel('reference', fontsize = 16)
#        plt.ylabel('predicted', fontsize = 16)
#        plt.tick_params(labelsize=14)
#        plt.scatter(y_test, mu_s)
#        plt.savefig(out_name + "_r2_" + str(i+1) + ".eps")
        
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
        if dataset == "OE" :
            fig_atom(df_62k, prediction_idxs, out_name + '_atoms_pre_' + str(i+1) +'.eps')
            fig_HOMO(homo_lowfid, prediction_idxs, out_name + '_HOMO_pre_' + str(i+1) + '.eps')
        elif dataset == "AA" :
            fig_HOMO(homo_lowfid, prediction_idxs, out_name + '_HOMO_pre_' + str(i+1) + '.eps')
            

        process_time_all = time.time() - start_all
        out_time_all(out_name, process_time_all)

        process_time_all_temp = time.time() - start_all_temp
        out_time_all_temp(out_name, process_time_all_temp)
        
    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)
    
    #-- score summary
    if num_itr > 0 :
        f = open(out_name, 'a')
        form="%i %20.10f %20.10f \n"
        for i in range(num_itr + 1):
            f.write(form % (i,r2_sum[i],MAE_sum[i]))
        f.close()    

    append_write(out_name,"End calculation !")

def str_to_bool(s):
        if s == 'True':
                     return True
        elif s == 'False':
                     return False
        else:
                     raise ValueError
            
#-- Start active_learning
if __name__ == "__main__":
        main()
