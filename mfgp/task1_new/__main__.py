import pdb
import sys
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
from io_utils import overwrite, append_write, out_condition, out_time, out_time_all, out_time_all_temp, fig_atom, fig_HOMO, fig_scatter_r2, fig_MDS_scatter_std, fig_MDS_scatter_label, Input
import datetime
import time
import multiprocessing as multi

def main():

    start_all = time.time()
    filepath = sys.argv[1]

    InData = Input(filepath)

    #-- Output
    out_name = InData.out_name
    #-- Condition
    fn_name = InData.fn_name
    K_high = InData.K_high
    rnd_size = InData.rnd_size
    random_seed = InData.random_seed
    #- Flag
    save_load_flag = InData.save_load_flag
    save_load_split_flag = InData.save_load_split_flag
    restart_flag = InData.restart_flag
    #-- Data set
    dataset = InData.dataset
    dataset_size = InData.dataset_size
    test_set_size = InData.test_set_size
    num_itr = InData.num_itr    
    pre_idxs = np.empty(int(num_itr+1),dtype = int)    
    for i in range(int(num_itr+1)):
        pre_idxs[i] = InData.pre_idxs[i]
    #-- Preprocess
    preprocess = InData.preprocess
    mbtr_red = InData.mbtr_red
    #-- Path 
    mbtr_path = InData.mbtr_path
    json_path = InData.json_path
    loadidxs_path = InData.loadidxs_path
    loadsplitidxs_path = InData.loadsplitidxs_path
    #-- Kernel
    kernel_type = InData.kernel_type
    length = InData.length
    const = InData.const
    bound = InData.bound
    n_opt = InData.n_opt

    # set the random seed
    np.random.seed(random_seed)

    #-- Initialize
    if restart_flag:
        append_write(out_name, "\n\n restart \n\n")
    else:
        overwrite(out_name, "")

    #-- Load for descriptor
    start = time.time()
    append_write(out_name, "start load mbtr \n")
    mbtr_data = load_npz(mbtr_path)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- Load for HOMO energy
    if dataset == "OE" :
        df_62k = 0
        start = time.time()	
        append_write(out_name, "start load OE \n")
        homo_lowfid = np.loadtxt(json_path)
        process_time = time.time() - start
        out_time(out_name, process_time)
        # start = time.time()
        # append_write(out_name, "start load df_62k \n")
        # df_62k = pd.read_json(json_path, orient='split')
        # process_time = time.time() - start
        # out_time(out_name, process_time)
        # 
        # num_atoms = df_62k["number_of_atoms"].values
        # 
        # homo_lowfid = df_62k.apply(lambda row: get_level(
        #     row, level_type='HOMO', subset='PBE+vdW_vacuum'),
        #                            axis=1).to_numpy()
    elif dataset == "AA":
        df_62k = 0
        start = time.time()
        append_write(out_name, "start load AA \n")
        homo_lowfid =  np.loadtxt(json_path)
        process_time = time.time() - start
        out_time(out_name, process_time)
    elif dataset == "QM9":
        df_62k = 0
        start = time.time()
        append_write(out_name, "start load QM9 \n")
        homo_lowfid =  np.loadtxt(json_path)
        process_time = time.time() - start
        out_time(out_name, process_time)
    else:
        append_write(out_name,"Dataset sould be AA or OE or QM9\n")
        append_write(out_name,"program stopped ! \n")    
        sys.exit()

    #-- Reduce the size of descriptor array
    if mbtr_red :
        mbtr_data_red = mbtr_data[:,mbtr_data.getnnz(0) > 0]
    else:
        mbtr_data_red = mbtr_data
        
    #-- Output
    out_condition(out_name, InData)
    f = open(out_name, 'a')
    f.write("mbtr_data col " + str(mbtr_data.shape[0]) + "\n" )
    f.write("mbtr_data row " + str(mbtr_data.shape[1]) + "\n" )
    mbtr_data_size = mbtr_data.shape[0]
    f.write("mbtr_data_size " + str(mbtr_data_size) + "\n" )
    f.write("mbtr_data_red_size " + str(mbtr_data_red.shape[1]) + "\n" )
    f.write("=============================" + "\n")
    append_write(out_name,str(0) + "-th learning \n")
    f.flush()
    f.close()

    #-- Figure of 62k dataset and AA dataset
    if dataset in ["OE", "QM9"] :
        # fig_atom(df_62k,range(61489),"atoms_all.eps")
        fig_HOMO(homo_lowfid,range(dataset_size), "HOMO_all.eps")
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
        kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) \
                 * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) # best result.    
    else:
        append_write(out_name,"kernel sould be RBF or constRBF \n")
        append_write(out_name,"program stopped ! \n")    
        sys.exit()
        
    normalize_y = False
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)
    append_write(out_name,"length of RBF kernel before fitting " + str(length) + "\n")
    append_write(out_name,"constant of constant kernel before fitting " + str(const) + "\n")

    #-- Save or Load index
    if save_load_flag == 'save':
        #-- Load prepared dataset
        if save_load_split_flag:
            append_write(out_name, "save_load_split_flag: True \n")
            load_idxs = np.load(loadsplitidxs_path)
            test_idxs = load_idxs['test_idxs']
            remaining_idxs = load_idxs['remaining_idxs']
            append_write(out_name, "1st index of test_idxs " + str(test_idxs[0]) + "\n" )
            append_write(out_name, "length of loaded split test_idxs " + str(len(test_idxs)) + "\n")
            append_write(out_name, "length of loaded split remaining_idxs " + str(len(remaining_idxs)) + "\n")
        #-- Not load prepared dataset, use original 62k dataset
        else:
            append_write(out_name, "save_load_split_flag: False \n")
            #-- Mbtr data size
            if dataset_size < mbtr_data_size:
                append_write(out_name, "mbtr_data_size =/ dataset_size \n")
                append_write(out_name, "mbtr_data_size " + str(mbtr_data_size) + "\n")
                append_write(out_name, "dataset_size " + str(dataset_size) + "\n")

                remaining_idxs, not_used_idxs = train_test_split(range(mbtr_data_size), train_size = dataset_size, random_state = random_seed)
                np.save(out_name + "_dataset_idxs", remaining_idxs)

                #-- Figure
                # if dataset == "OE" :
                #     # fig_atom(df_62k,remaining_idxs, out_name +'_atoms_pre_dataset.eps')
                #     fig_HOMO(homo_lowfid,remaining_idxs, out_name + 'HOMO_pre_dataset.eps')
                # elif dataset == "AA" :
                #     fig_HOMO(homo_lowfid,remaining_idxs, out_name + 'HOMO_pre_dataset.eps')
                if dataset in ["OE", "AA", "QM9"]:
                    print(f"Dataset {dataset} homo_lowfig {len(homo_lowfid)}, Remaining idxs {len(remaining_idxs)}")
                    print(f"Remaining idxs max {max(remaining_idxs)} min {min(remaining_idxs)}")
                    fig_HOMO(homo_lowfid,remaining_idxs, out_name + 'HOMO_pre_dataset.eps')

                #-- test , 1st train, remaining split.
                test_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = test_set_size, random_state = random_seed)
                np.savez(out_name + "_dataset_split_idxs.npz", remaining_idxs=remaining_idxs, test_idxs=test_idxs)

            elif dataset_size  == mbtr_data_size:    
                append_write(out_name, "mbtr_data_size ==  dataset_size \n")
                test_idxs, remaining_idxs = train_test_split(range(mbtr_data_size), train_size = test_set_size, random_state = random_seed)

            else: 
                append_write(out_name,"dataset_size should be = or < mbtr_data_size \n")
                append_write(out_name,"program stopped ! \n")
                sys.exit()

            append_write(out_name, "length of test_idxs " + str(len(test_idxs)) + "\n")
            append_write(out_name, "length of remaining_idxs " + str(len(remaining_idxs)) + "\n")            

        prediction_set_size = pre_idxs[0]
        prediction_idxs, remaining_idxs = pre_rem_split(prediction_set_size, remaining_idxs, random_seed)
        append_write(out_name, "length of prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name, "length of remaining_idxs " + str(len(remaining_idxs)) + "\n")

        #-- Save the values
        append_write(out_name, "save_load_flag: save \n")
        np.savez(out_name + "_0_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, test_idxs=test_idxs)

    #-- Load index
    elif save_load_flag =='load':
        append_write(out_name, "save_load_flag: load \n")
        load_idxs = np.load(loadidxs_path)
        remaining_idxs = load_idxs['remaining_idxs']
        prediction_idxs = load_idxs['prediction_idxs']
        test_idxs = load_idxs['test_idxs']
        append_write(out_name, "1st index of test_idxs " + str(test_idxs[0]) + "\n")
        append_write(out_name, "length of loaded test_idxs " + str(len(test_idxs)) + "\n")
        append_write(out_name, "length of loaded prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name, "length of loaded remaining_idxs " + str(len(remaining_idxs)) + "\n")
        
    #-- Save_load_flag should be "save" or "load"
    else:
        append_write(out_name,"save_load_flag should be save or load \n")
        append_write(out_name,"program stopped ! \n")
        sys.exit()
        
    #-- 
    X_train, X_test = mbtr_data_red[prediction_idxs, :], mbtr_data_red[test_idxs, :]
    y_train, y_test = homo_lowfid[prediction_idxs], homo_lowfid[test_idxs]

    #-- Preprocessing
    X_train_pp, X_test_pp = desc_pp(preprocess, X_train, X_test)

    #---------------------------------------------
    #-- Fit the model
    # sample_sum = np.array([])
    # r2_sum = np.array([])
    # MAE_sum = np.array([])
    sample_sum = 0
    r2_sum = 0
    MAE_sum = 0
    length_sum = 0 
    const_sum = 0
    i = -1

    ## if the following files exist, load the data into the variable
    ## and prepend it.
    get_path = lambda file_suffix: Path(out_name + file_suffix + ".npy")
    def exists(file_suffix):
        file = get_path(file_suffix) 
        return file.is_file()

    # def get_data(file_suffix):
    #     filename = out_name + file_suffix + ".npy"
    #     print(f"loading data from {filename}")
    #     data = np.load(filename)
    #     return data

    filename_var_dict = {"_sample": sample_sum, "_r2" : r2_sum, "_MAE" : MAE_sum}
    # 
    # for file_suffix in filename_var_dict.keys():
    #     var = filename_var_dict[file_suffix]
    #     if exists(file_suffix):
    #         var = get_data(file_suffix)
    #         print(f"Loaded : {file_suffix} , contents : {var}")
    # 
    # print(f"DEBUG : MAE_sum in the beginning : {MAE_sum}")

    for file_suffix in filename_var_dict.keys():
        var = filename_var_dict[file_suffix]
        if exists(file_suffix):
            p = get_path(file_suffix)
            fname_noextension = p.stem
            fname_ext = p.suffix
            datestr = datetime.datetime.today().strftime("%Y-%m-%dT%I-%M-%S")
            new_fname = fname_noextension + "_" + datestr + fname_ext
            new_path = Path(p.parent, new_fname)
            p.rename(new_path)
            print(f"Renamed {p} to {new_path}")


    sample_sum,r2_sum,MAE_sum,length_sum,const_sum,const,length = main_loop(i,out_name,dataset,X_train_pp,X_test_pp,y_train,gpr,kernel_type,bound,const_sum,length_sum,y_test,prediction_idxs,sample_sum,r2_sum,MAE_sum,homo_lowfid,df_62k, kernel,normalize_y,n_opt,random_seed)
    #---------------------------------------------        

    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)

    should_i_resume = False
    #-- Iteration start
    for i in range(num_itr):
        append_write(out_name,"============================= \n")
        append_write(out_name,str(i+1) + "-th learning" + "\n")
        print(f"{out_name} {str(i+1)}-th learning")
        start_all_temp = time.time()
        
        # pdb.set_trace()
        # check if this iteration has already been done.
        saved_idxs_file = Path(out_name + "_" + str(i+1) + "_full_idxs.npz")
        saved_params_file = Path(out_name + "_para_kernel_aft_" + str(i+1) + ".txt")
        if saved_params_file.is_file():
            # i.e. if the (i+1)th saved param file exists then
            # we can continue.
            should_i_resume = True
            continue
        else:
            # (i+1)th saved param file doesn't exit, which means
            # (i)th saved param file is the last one.
            if i > 0 and should_i_resume:
                should_i_resume = False
                print(f"loading {out_name}_para_kernel_aft_{str(i)}.txt")
                append_write(out_name, f"loading {out_name}_para_kernel_aft_{str(i)}.txt")
                with open(f"{out_name}_para_kernel_aft_{str(i)}.txt") as f:
                    data = eval(f.read())
                    gpr = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)            
                    # gpr.kernel.k1.constant_value = data['k1__constant_value']
                    # gpr.kernel.k2.length_scale = data['k2__length_scale']

                    # gpr2 = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)            
                    # gpr2.kernel.k1.constant_value = data['k1__constant_value']
                    # gpr2.kernel.k2.length_scale = data['k2__length_scale']
                    print(f"Inside loading section : {gpr.get_params}")
                # also load the data
                print(f"loading {out_name}_{str(i+1)}_full_idxs.npz")
                with np.load(out_name + "_" + str(i+1) + "_full_idxs.npz") as data:
                    prediction_idxs = data['prediction_idxs']
                    remaining_idxs = data['remaining_idxs']
                    test_idxs = data['test_idxs']
                # refit GPR with these idxs because that's used in acq_fn to pick next points.
                X_train_pp, _ = desc_pp(preprocess, mbtr_data_red[prediction_idxs, :], X_test)
                y_train = homo_lowfid[prediction_idxs]
                gpr.fit(X_train_pp, y_train)
                # # gpr2.fit(X_train_pp, y_train)
                # gpr2.log_marginal_likelihood_value_ = -1040.929968135276
                # gpr2._y_train_mean = np.array([0.])
                # gpr2.X_train_ = X_train_pp
                print(f"GPR vals : k1.constant_value {gpr.kernel.k1.constant_value} k2.length_scale {gpr.kernel.k2.length_scale}")
                # print(f"GPR2 vals : k1.constant_value {gpr2.kernel.k1.constant_value} k2.length_scale {gpr2.kernel.k2.length_scale}")
                # pdb.set_trace()
                # gpr=gpr2

        # if saved_idxs_file.is_file():
        #     # file exists, skip this iteration
        #     append_write(out_name," Iteration already done skipping to the next one.")
        #     should_i_resume = True
        #     continue
        # else:
        #     # file doesn't exist load the prediction_idxs, remaining_idxs and
        #     # test_idxs from the previous iterations's file. If the iteration is 0
        #     # this means this is the first run, nothing to load.
        #     if i > 0 and should_i_resume:
        #         should_i_resume = False # since we have already resumed.
        #         print(f"loading {out_name}_{str(i)}_full_idxs.npz")
        #         append_write(out_name, f"loading {out_name}_{str(i)}_full_idxs.npz")
        #         with np.load(out_name + "_" + str(i) + "_full_idxs.npz") as data:
        #             prediction_idxs = data['prediction_idxs']
        #             remaining_idxs = data['remaining_idxs']
        #             test_idxs = data['test_idxs']
        #         # refit GPR with these idxs because that's used in acq_fn to pick next points.
        #         X_train_pp, _ = desc_pp(preprocess, mbtr_data_red[prediction_idxs, :], X_test)
        #         y_train = homo_lowfid[prediction_idxs]
        #         gpr = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)            
        #         ## Reload hyperparams
        #         with open(out_name + '_para_kernel_aft_' + str(i-1) + '.txt') as f2:
        #             data = eval(f2.read())
        #         pdb.set_trace()
        #         gpr.kernel.k1.constant_value = data['k1__constant_value']
        #         gpr.kernel.k2.length_scale = data['k2__length_scale']
        #         print(f"Inside loading section : {gpr.get_params}")
                
                # # pdb.set_trace()
                # ## gpr.fit(X_train_pp, y_train)
                # param_file_pkl = Path(out_name + "_para_kernel_aft_" + str(i-1) + ".pkl")
                # # if param_file.is_file():
                # #     with open(out_name + '_para_kernel_aft_' + str(i+1) + '.txt') as f2:
                # #         params = eval(f2.read())
                # #         gpr.set_params(**params)
                # #         print(f"Loaded gpr params from {param_file}")
                # #         para_kernel_aft = params
                # if param_file_pkl.is_file():
                #     print(f"LOADING PARAM FILE..... idx : {i}")
                #     with param_file_pkl.open("rb") as f:
                #         data = pkl.load(f)
                #         print(f"loading params : {data}")
                #         gpr = gpr.set_params(**data)
                #         with open(out_name + '_para_kernel_aft_' + str(i-1) + '.txt') as f2:
                #             data = eval(f2.read())
                #         pdb.set_trace()
                #         gpr.kernel.k1.constant_value = data['k1__constant_value']
                #         gpr.kernel.k2.length_scale = data['k2__length_scale']
                #         # gpr.kernel.set_params(**data)
                #         # gpr = gpr.kernel.set_params(**data)
                #         # HERE, the kernel hyper params are not loaded correctly.
                #         # pdb.set_trace()
                #         print(f"Inside loading section : {gpr.get_params}")


        prediction_set_size = pre_idxs[i+1]

        # print(f"Prediciton idxs {i}-before : {prediction_idxs}")
        prediction_idxs, remaining_idxs, X_train_pp, y_train = acq_fn(fn_name\
                                                                      , i\
                                                                      , prediction_idxs\
                                                                      , remaining_idxs\
                                                                      , prediction_set_size\
                                                                      , rnd_size\
                                                                      , mbtr_data_red\
                                                                      , homo_lowfid\
                                                                      , K_high\
                                                                      , gpr\
                                                                      , preprocess\
                                                                      , out_name\
                                                                      , random_seed)
        # print(f"Prediciton idxs {i}-after : {prediction_idxs}")

        #-- Save the values 
        np.savez(out_name + "_" + str(i+1) + "_full_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs = prediction_idxs, test_idxs = test_idxs)

        append_write(out_name,"check duplicates of prediction_idxs " + str(has_duplicates(prediction_idxs)) + "\n")
        all_idxs = np.concatenate([test_idxs, prediction_idxs, remaining_idxs], 0)
        append_write(out_name,"length of all_idxs " + str(len(all_idxs)) + "\n")
        append_write(out_name,"check duplicates of all idxs " + str(has_duplicates(all_idxs)) + "\n")
        append_write(out_name,"length of prediction_idxs " + str(len(prediction_idxs)) + "\n")
        append_write(out_name,"length of remaining_idxs " + str(len(remaining_idxs)) + "\n")
        append_write(out_name,"length of test_idxs " + str(len(test_idxs)) + "\n")

        gpr = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)            
        append_write(out_name,"length of RBF kernel before fitting " + str(length) + "\n")
        append_write(out_name,"constant of constant kernel before fitting " + str(const) + "\n")
        #---------------------------------------------
        #-- Fit the model
        sample_sum,r2_sum,MAE_sum,length_sum,const_sum,const,length = main_loop(i,out_name,dataset,X_train_pp,X_test_pp,y_train,gpr,kernel_type,bound,const_sum,length_sum,y_test,prediction_idxs,sample_sum,r2_sum,MAE_sum,homo_lowfid,df_62k, kernel, normalize_y, n_opt, random_seed)
        #---------------------------------------------            

        process_time_all = time.time() - start_all
        out_time_all(out_name, process_time_all)

        process_time_all_temp = time.time() - start_all_temp
        out_time_all_temp(out_name, process_time_all_temp)
        


    process_time_all = time.time() - start_all
    out_time_all(out_name, process_time_all)
    
    #-- Score summary
    if num_itr > 0 :
        #-- Output score
        append_write(out_name, "\n Output for score \n")
        append_write(out_name, "Times     r2                 MAE \n")        
        f = open(out_name, 'a')
        form="%i %20.10f %20.10f \n"
        for i in range(num_itr + 1):
            # pdb.set_trace()
            f.write(form % (i,r2_sum[i],MAE_sum[i]))
        f.close()    
        append_write(out_name, "\n Output for hyper parameters \n")
        append_write(out_name, "times    const              length \n")                
        #-- Output hypterparamters
        f = open(out_name, 'a')
        form="%i %20.10f %20.10f \n"
        for i in range(num_itr + 1):
            f.write(form % (i,const_sum[i],length_sum[i]))
        f.close()    
        
    append_write(out_name,"End calculation !")

    
def main_loop(i,out_name,dataset,X_train_pp,X_test_pp,y_train,gpr,kernel_type,bound,const_sum,length_sum,y_test,prediction_idxs,sample_sum,r2_sum,MAE_sum,homo_lowfid,df_62k, kernel,normalize_y,n_opt,random_seed):
    print(f"DEBUG : Inside main_loop , value of MAE_sum is {MAE_sum}")

    #-- Fit the model
    append_write(out_name,"initial learning \n")
    start = time.time()
    append_write(out_name,"start to make a model \n")
    print(X_train_pp)
    # gpr = GaussianProcessRegressor( kernel = kernel, normalize_y = normalize_y, n_restarts_optimizer = n_opt, random_state = random_seed)            
    gpr.fit(X_train_pp, y_train)
    process_time = time.time() - start
    out_time(out_name, process_time)

    # pdb.set_trace()
    #-- Load hyper parameter if file exists
    param_file = Path(out_name + "_para_kernel_aft_" + str(i+1) + ".txt")
    param_file_pkl = Path(out_name + "_para_kernel_aft_" + str(i+1) + ".pkl")
    # if param_file.is_file():
    #     with open(out_name + '_para_kernel_aft_' + str(i+1) + '.txt') as f2:
    #         params = eval(f2.read())
    #         gpr.set_params(**params)
    #         print(f"Loaded gpr params from {param_file}")
    #         para_kernel_aft = params
    if param_file_pkl.is_file():
        with param_file_pkl.open("rb") as f:
            data = pkl.load(f)
            # pdb.set_trace()
            print(f"loading params : {data}")
            gpr = gpr.set_params(**data)
            para_kernel_aft = gpr.kernel_.get_params()
    else:
        #-- Save hyper paramter
        para_kernel_aft = gpr.kernel_.get_params()
        with open(out_name + '_para_kernel_aft_' + str(i+1) + '.txt', 'w') as f2:
            print(para_kernel_aft, file=f2)

        para_aft = gpr.get_params()
        with open(out_name + '_para_kernel_aft_' + str(i+1) + ".pkl", "wb") as f:
            print(f"saving params : {para_aft}")
            pkl.dump(para_aft, f)

   #-- Get hyper paramter
    if kernel_type == "RBF":
        length = para_kernel_aft['length_scale']
        kernel = RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound))
    elif kernel_type == "constRBF":
        const = para_kernel_aft['k1__constant_value']
        length = para_kernel_aft['k2__length_scale']
        kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) \
                 * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) 

    #-- Save hyper paramters
    if i == -1 :
        const_sum = const
        length_sum = length
    elif i > -1 :
        const_sum = np.append(const_sum ,const)
        length_sum = np.append(length_sum, length)
        
    #-- Predict
    start = time.time()
    append_write(out_name,"start prediction" + "\n")
    print(f"Iteration {i} : gpr params {gpr.get_params()} . k1_consant_value {const} k2_length_scale {length}")
    mu_s, std_s = gpr.predict(X_test_pp, return_std=True)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- Score(r^2 by hand)
    start = time.time()
    append_write(out_name,"start calculating r2 by hand \n")
    r2 = r2_byhand(y_test, mu_s)
    process_time = time.time() - start
    out_time(out_name, process_time)

    #-- Score(MAE)
    start = time.time()
    append_write(out_name,"start calculating MAE \n")
    MAE = np.array(mean_absolute_error(y_test, mu_s))
    process_time = time.time() - start
    out_time(out_name, process_time)

    append_write(out_name,"r2 by hand " + str(r2) + "\n")
    append_write(out_name,"MAE " + str(MAE) + "\n")
    
    #-- Plot r2
    fig_scatter_r2(y_test, mu_s, out_name + '_r2_' + str(i+1) + '.eps')

    #-- Score array
    # if MAE_sum.size == 0: 
    if i == -1:
        sample_sum = np.array(len(prediction_idxs))
        r2_sum = r2
        MAE_sum = MAE
    elif i > -1:
        sample_sum = np.append(sample_sum ,len(prediction_idxs))
        r2_sum = np.append(r2_sum ,r2)        
        MAE_sum = np.append(MAE_sum, MAE)
    # pdb.set_trace()
    # sample_sum = np.append(sample_sum ,len(prediction_idxs))
    # r2_sum = np.append(r2_sum ,r2)        
    # MAE_sum = np.append(MAE_sum, MAE)
    
        
    #-- Save score
    # pdb.set_trace()
    np.save(out_name + "_sample", sample_sum)
    np.save(out_name + "_r2", r2_sum)    
    np.save(out_name + "_MAE", MAE_sum)
    np.savez(out_name + "_score.npz", sample=sample_sum, r2=r2_sum,MAE=MAE_sum)
    
    #-- Save mean and std of test data
    np.save(out_name + "_mu_" + str(i+1),mu_s)
    np.save(out_name + "_std_" + str(i+1) ,std_s)
    
    #-- Figure
    # if dataset == "OE" :
    #     # fig_atom(df_62k, prediction_idxs, out_name +'_atoms_pre_' + str(i+1) + '.eps')
    #     fig_HOMO(homo_lowfid, prediction_idxs, out_name + 'HOMO_pre_' + str(i+1) + '.eps')
    # elif dataset == "AA" :
    #     fig_HOMO(homo_lowfid, prediction_idxs, out_name + 'HOMO_pre_'+  str(i+1) + '.eps')
    if dataset in ["OE", "AA", "QM9"]:
        fig_HOMO(homo_lowfid, prediction_idxs, out_name + 'HOMO_pre_'+  str(i+1) + '.eps')

    return sample_sum,r2_sum,MAE_sum,length_sum,const_sum,const,length

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
