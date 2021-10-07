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
from sklearn.externals import joblib

def get_data_given_indices(conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid):
    # get the data
    append_write(conf.out_name, f"Getting data corresponding to the latest indices.")
    X_train, X_test = mbtr_data_red[pred_idxs, :], mbtr_data_red[test_idxs, :]
    y_train, y_test = homo_lowfid[pred_idxs], homo_lowfid[test_idxs]
    X_train_pp, X_test_pp = desc_pp(conf.preprocess, X_train, X_test)
    return X_train_pp, X_test_pp, y_train, y_test

def get_gpr_params(gpr):
  try:
    params = gpr.kernel_.get_params() 
  except AttributeError as e:
    params = gpr.kernel.get_params()
  const  = params['k1__constant_value']
  length = params['k2__length_scale']
  return const, length

def exists(file):
  file = Path(file) 
  return file.is_file()

# def compute_stats(out_name, preprocess, gpr, X_test_pp, y_test):
def compute_stats(conf, gpr, X_test_pp, y_test):
  out_name = conf.out_name
  start = time.time()
  append_write(conf.out_name, "starting prediction \n")
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

def get_gpr(out_name, const, bound, length, kernel_type, normalize_y, n_opt, random_seed, alpha):
  if kernel_type == "RBF":
      kernel = RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound))
  elif kernel_type == "constRBF":
      kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) \
               * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) # best result.    
  else:
      append_write(out_name,"kernel sould be RBF or constRBF \n")
      append_write(out_name,"program stopped ! \n")    
      sys.exit()
      
  # normalize_y = normalize_y
  gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, 
                                 n_restarts_optimizer = n_opt, 
                                 random_state = random_seed, alpha=alpha)
  append_write(out_name,"length of RBF kernel before fitting " + str(length) + "\n")
  append_write(out_name,"constant of constant kernel before fitting " + str(const) + "\n")
  return gpr

def main():
  start_all = time.time()
  filepath = sys.argv[1]
  #>>> Load Config
  conf = Input(filepath)

  # #-- Output
  # out_name = InData.out_name
  # #-- Condition
  # fn_name = InData.fn_name
  # K_high = InData.K_high
  # rnd_size = InData.rnd_size
  # random_seed = InData.random_seed
  # #- Flag
  # save_load_flag = InData.save_load_flag
  # save_load_split_flag = InData.save_load_split_flag
  # restart_flag = InData.restart_flag
  # #-- Data set
  # dataset = InData.dataset
  # dataset_size = InData.dataset_size
  # test_set_size = InData.test_set_size
  # num_itr = InData.num_itr    
  # pre_idxs = np.empty(int(num_itr+1),dtype = int)    
  # for i in range(int(num_itr+1)):
  #   pre_idxs[i] = InData.pre_idxs[i]
  # #-- Preprocess
  # preprocess = InData.preprocess
  # mbtr_red = InData.mbtr_red
  # #-- Path 
  # mbtr_path = InData.mbtr_path
  # json_path = InData.json_path
  # loadidxs_path = InData.loadidxs_path
  # loadsplitidxs_path = InData.loadsplitidxs_path
  # #-- Kernel
  # kernel_type = InData.kernel_type
  # length = InData.length
  # const = InData.const
  # bound = InData.bound
  # n_opt = InData.n_opt
  # alpha = InData.alpha
  # normalize_y = InData.normalize_y
  # #<<< End Loading config.

  # set the random seed
  np.random.seed(conf.random_seed)

  # #-- Initialize
  # if restart_flag:
  #   append_write(out_name, "\n\n restart \n\n")
  # else:
  #   overwrite(out_name, "")

  append_write(conf.out_name, datetime.datetime.today().strftime("%Y-%m-%dT%I-%M-%S"))
  #>>> Load Dataset
  #-- Load for descriptor
  start = time.time()
  append_write(conf.out_name, "start load mbtr \n")
  mbtr_data = load_npz(conf.mbtr_path)
  process_time = time.time() - start
  out_time(conf.out_name, process_time)

  #-- Load for HOMO energy
  if conf.dataset in ["OE", "AA", "QM9"]:
    df_62k = 0
    start = time.time()	
    append_write(conf.out_name, "start load OE \n")
    homo_lowfid = np.loadtxt(conf.json_path)
    process_time = time.time() - start
    out_time(conf.out_name, process_time)
  else:
    append_write(conf.out_name,"Dataset sould be AA or OE or QM9\n")
    append_write(conf.out_name,"program stopped ! \n")    
    sys.exit()

  #-- Reduce the size of descriptor array
  if conf.mbtr_red :
    mbtr_data_red = mbtr_data[:,mbtr_data.getnnz(0) > 0]
  else:
    mbtr_data_red = mbtr_data

  #-- Output
  out_condition(conf.out_name, conf)
  f = open(conf.out_name, 'a')
  f.write("mbtr_data col " + str(mbtr_data.shape[0]) + "\n" )
  f.write("mbtr_data row " + str(mbtr_data.shape[1]) + "\n" )
  mbtr_data_size = mbtr_data.shape[0]
  f.write("mbtr_data_size " + str(mbtr_data_size) + "\n" )
  f.write("mbtr_data_red_size " + str(mbtr_data_red.shape[1]) + "\n" )
  f.write("=============================" + "\n")
  append_write(conf.out_name,str(0) + "-th learning \n")
  f.flush()
  f.close()

  #-- Figure of 62k dataset and AA dataset
  if conf.dataset in ["OE", "QM9", "AA"] :
      # fig_atom(df_62k,range(61489),"atoms_all.eps")
      fig_HOMO(homo_lowfid,range(conf.dataset_size), "HOMO_all.eps")
  else:
      append_write(conf.out_name,"Dataset sould be AA, OE or QM9 \n")
      append_write(conf.out_name,"program stopped ! \n")    
      sys.exit()
  #<<< End loading data

  #>>> Setup the GP model
  n_features = int(mbtr_data_red.shape[1])
  gpr = get_gpr(conf.out_name, conf.const, conf.bound, conf.length, conf.kernel_type, conf.normalize_y, conf.n_opt, conf.random_seed, conf.alpha)
  #<<< Finished setting up GP model


  # if the out_name file exists rename to a different file so
  # contents are not overwritten.
  # ------------ No need cause we are appending.
  # if exists(out_name):
  #     p = Path(out_name)
  #     datestr = datetime.datetime.today().strftime("%Y-%m-%dT%I-%M-%S")
  #     new_fname = p.stem + "_" + datestr + p.suffix
  #     new_path = Path(p.parent ,new_fname)
  #     p.rename(new_path)
  #     print(f"Renamed {p} to {new_path}")


  ####################################################
  #       How active learning training should        #
  #                  proceed.                        #
  ####################################################
  #
  # Total number of iterations = len(pre_idxs)
  # 0. Get rem, test = train_test_plit(rem); iter = 0
  # 1. split dataset pre, rem = pre_rem_split(number = pre_idxs[0]); idxs_0 = [pre, test, rem]
  # 2. iter = iter + 1 
  # 3. train GP on pre
  # 4. Save the GP model using pickle
  # 5. Use GP from 3. in acq_func and pick pre_idxs[1] number of molecules. call them pre_new
  # 6. rem = rem - pre_new; pre = pre_new; save in _{iter}_full_idxs.npz
  # 7. Goto 2.

  #>>> Identify indices file from last training iteration.
  for idx, batch_size in enumerate(conf.pre_idxs[1:], 1):
    # Try to open the full_idxs file, if you can, save the last rem, train, test.
    # If you cannot, use the latest rem train test indices to continue.
    try:
      append_write(conf.out_name, f"{idx}, {batch_size}\n")
      data = np.load(f"{conf.out_name}_{idx}_full_idxs.npz")
      rem_idxs  = data['remaining_idxs']
      pred_idxs = data['prediction_idxs']
      test_idxs = data['test_idxs']
      # pdb.set_trace()
      print("Loaded data.")
      append_write(conf.out_name, f"loaded {conf.out_name}_{idx}_full_idxs.npz Continuing iteration.\n")
    except Exception as e:
      append_write(conf.out_name, f"iteration {idx} -- {e}, Couldn't load _full_idxs for index {idx}\n")
      break
      # So we don't have the full_idxs file corresponding to the latest idx.
      # the previous rem_idxs, pred_idxs, test_idxs are already loaded.


  # Now try to load the model file corresponding to idx-1
  # Since we will need that when we run the acquition function for the next index.
  idx = idx - 1 
  try:
    # try to load gpr
    gpr = joblib.load(f"{conf.out_name}_{idx}_model.pkl")
    append_write(conf.out_name, f"Loaded {conf.out_name}_{idx}_model.pkl \n")
  except Exception as e:
    # if can't load train again
    append_write(conf.out_name, f"Can't load {conf.out_name}_{idx}_model.pkl, retraining \n")
    X_train_pp, X_test_pp, y_train, y_test = get_data_given_indices(conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid)
    gpr.fit(X_train_pp, y_train)
    # save GP model
    joblib.dump(gpr, f"{conf.out_name}_{idx}_model.pkl")
    append_write(conf.out_name, f"Trained {conf.out_name}_{idx}_model.pkl and saved it to disk")

  # Now we can resume from the next index
  idx = idx + 1
  for idx, batch_size in enumerate(conf.pre_idxs[idx:], idx):
    append_write(conf.out_name, f"Resuming from index {idx} and current batch size is {batch_size}\n")
    print(idx, batch_size)
    # pdb.set_trace()
    start = time.time()
    # get the data
    # X_train, X_test = mbtr_data_red[pred_idxs, :], mbtr_data_red[test_idxs, :]
    # y_train, y_test = homo_lowfid[pred_idxs], homo_lowfid[test_idxs]

    # train GP with pred_idxs
    # X_train_pp, X_test_pp = desc_pp(conf.preprocess, X_train, X_test)
    ## X_train_pp, X_test_pp, y_train, y_test = get_data_given_indices(conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid):

    const, length = get_gpr_params(gpr) 
    append_write(conf.out_name, f"length of RBF kernel before fitting {length} \n")
    append_write(conf.out_name, f"constant of constant kernel before fitting {const} \n")

    # # pdb.set_trace()
    # try:
    #   # try to load gpr
    #   gpr = joblib.load(f"{conf.out_name}_{idx}_model.pkl")
    #   append_write(conf.out_name, f"Loaded {conf.out_name}_{idx}_model.pkl \n")
    # except Exception as e:
    #   # if can't load train again
    #   append_write(conf.out_name, f"Can't load {conf.out_name}_{idx}_model.pkl, retraining \n")
    #   gpr.fit(X_train_pp, y_train)
    #   # save GP model
    #   joblib.dump(gpr, f"{conf.out_name}_{idx}_model.pkl")

    # const, length = get_gpr_params(gpr) 
    # append_write(conf.out_name, f"length of RBF kernel after fitting {length} \n")
    # append_write(conf.out_name, f"constant of constant kernel after fitting {const} \n")

    append_write(conf.out_name, "Finished training \n")
    process_time = time.time() - start
    out_time(conf.out_name, process_time)

    # Go through acq_fn to get new pred_idxs
    prediction_set_size = batch_size
    start = time.time()
    append_write(conf.out_name, f"Starting acq function.\n")
    pred_idxs, rem_idxs, X_train_pp, y_train = acq_fn(conf.fn_name\
                                                      , idx\
                                                      , pred_idxs\
                                                      , rem_idxs\
                                                      , prediction_set_size\
                                                      , conf.rnd_size\
                                                      , mbtr_data_red\
                                                      , homo_lowfid\
                                                      , conf.K_high\
                                                      , gpr\
                                                      , conf.preprocess\
                                                      , conf.out_name\
                                                      , conf.random_seed)
    append_write(conf.out_name, f"Acq fun done.\n")
    process_time = time.time() - start 
    out_time(conf.out_name, process_time)
    print(f"Saving {idx} _full_idxs file.")
    np.savez(f"{conf.out_name}_{idx}_full_idxs.npz",remaining_idxs=rem_idxs, prediction_idxs = pred_idxs, test_idxs = test_idxs)
    X_train_pp, X_test_pp, y_train, y_test = get_data_given_indices(conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid)
    append_write(conf.out_name, f"Training a GP model for index {idx}")
    gpr.fit(X_train_pp, y_train)
    # save GP model
    joblib.dump(gpr, f"{conf.out_name}_{idx}_model.pkl")
    append_write(conf.out_name, f"Trained {conf.out_name}_{idx}_model.pkl and saved it to disk")
    compute_stats(conf, gpr, X_test_pp, y_test)
  #<<<

if __name__=="__main__":
  main()
