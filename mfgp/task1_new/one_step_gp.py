from datetime import datetime
from pathlib import Path
from sklearn.externals import joblib
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.metrics import mean_absolute_error
from acq import acq_fn
from utils import desc_pp, desc_pp_notest, has_duplicates, pre_rem_split, r2_byhand
from io_utils import (
    append_write,
    out_condition,
    out_time,
    fig_HOMO,
    Input,
)

import sys
import numpy as np
import time
import matplotlib

matplotlib.use("Agg")


def get_data_from_last_training_run(conf):
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
    # >>> Identify indices file from last training iteration.
    for idx, batch_size in enumerate(conf.pre_idxs[1:], 1):
        # Try to open the full_idxs file, if you can, save the last rem, train, test.
        # If you cannot, use the latest rem train test indices to continue.
        try:
            append_write(conf.out_name, f"{idx}, {batch_size}\n")
            data = np.load(f"{conf.out_name}_{idx}_full_idxs.npz")
            rem_idxs = data["remaining_idxs"]
            pred_idxs = data["prediction_idxs"]
            test_idxs = data["test_idxs"]
            last_loaded_index = idx
            print("Loaded data.")
            append_write(
                conf.out_name,
                f"loaded {conf.out_name}_{idx}_full_idxs.npz Continuing iteration.\n",
            )
        except Exception as e:
            append_write(
                conf.out_name,
                f"iteration {idx} -- {e}, Couldn't load _full_idxs for index {idx}\n",
            )
            break
            # So we don't have the full_idxs file corresponding to the latest idx.
            # the previous rem_idxs, pred_idxs, test_idxs are already loaded.
    return rem_idxs, pred_idxs, test_idxs, last_loaded_index


def get_data_given_indices(conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid):
    # get the data
    append_write(conf.out_name, f"Getting data corresponding to the latest indices")
    x_train, x_test = mbtr_data_red[pred_idxs, :], mbtr_data_red[test_idxs, :]
    y_train, y_test = homo_lowfid[pred_idxs], homo_lowfid[test_idxs]
    x_train_pp, x_test_pp = desc_pp(conf.preprocess, x_train, x_test)
    return x_train_pp, x_test_pp, y_train, y_test


def get_gp_model(conf, idx, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid):
    """
    Loads an existing GP model given the index.
    If the model doesn't exist, gets the data corresponding to the index,
    Trains the GP model and returns that.
    """
    # >>> Setup the GP model
    gpr = get_gpr(
        conf.out_name,
        conf.const,
        conf.bound,
        conf.length,
        conf.kernel_type,
        conf.normalize_y,
        conf.n_opt,
        conf.random_seed,
        conf.alpha,
    )
    # <<< Finished setting up GP model

    try:
        # try to load gpr
        gpr = joblib.load(f"{conf.out_name}_{idx}_model.pkl")
        append_write(conf.out_name, f"Loaded {conf.out_name}_{idx}_model.pkl\n")
    except Exception:
        # if can't load train again
        append_write(
            conf.out_name, f"Can't load {conf.out_name}_{idx}_model.pkl, retraining\n"
        )
        x_train_pp, _, y_train, _ = get_data_given_indices(
            conf, pred_idxs, test_idxs, mbtr_data_red, homo_lowfid
        )
        gpr.fit(x_train_pp, y_train)
        # save GP model
        joblib.dump(gpr, f"{conf.out_name}_{idx}_model.pkl")
        append_write(
            conf.out_name,
            f"Trained {conf.out_name}_{idx}_model.pkl and saved it to disk",
        )
    return gpr


def get_gpr_params(gpr):
    try:
        params = gpr.kernel_.get_params()
    except AttributeError:
        params = gpr.kernel.get_params()
    const = params["k1__constant_value"]
    length = params["k2__length_scale"]
    return const, length


def exists(file):
    file = Path(file)
    return file.is_file()


# def compute_stats(out_name, preprocess, gpr, x_test_pp, y_test):
def compute_stats(conf, gpr, x_test_pp, y_test):
    out_name = conf.out_name
    start = time.time()
    append_write(conf.out_name, "starting prediction \n")
    mu_s, _ = gpr.predict(x_test_pp, return_std=True)
    process_time = time.time() - start
    out_time(out_name, process_time)

    # -- Score(r^2 by hand)
    start = time.time()
    append_write(out_name, "start calculating r2 by hand \n")
    r2 = r2_byhand(y_test, mu_s)
    process_time = time.time() - start
    out_time(out_name, process_time)

    # -- Score(MAE)
    start = time.time()
    append_write(out_name, "start calculating MAE \n")
    mae = np.array(mean_absolute_error(y_test, mu_s))
    process_time = time.time() - start
    out_time(out_name, process_time)

    append_write(out_name, f"r2 by hand {r2}\n")
    append_write(out_name, f"MAE {mae} \n")


def get_gpr(
    out_name, const, bound, length, kernel_type, normalize_y, n_opt, random_seed, alpha
):
    if kernel_type == "RBF":
        kernel = RBF(
            length_scale=length,
            length_scale_bounds=(length * 1.0 / bound, length * bound),
        )
    elif kernel_type == "constRBF":
        kernel = ConstantKernel(
            constant_value=const,
            constant_value_bounds=(const * 1.0 / bound, const * bound),
        ) * RBF(
            length_scale=length,
            length_scale_bounds=(length * 1.0 / bound, length * bound),
        )  # best result
    else:
        append_write(out_name, "kernel sould be RBF or constRBF \n")
        append_write(out_name, "program stopped ! \n")
        sys.exit()

    # normalize_y = normalize_y
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_opt,
        random_state=random_seed,
        alpha=alpha,
    )
    append_write(out_name, f"length of RBF kernel before fitting {length}\n")
    append_write(out_name, f"constant before fitting {const}\n")
    return gpr


def load_data(conf):
    # -- Load for HOMO energy
    if conf.dataset in ["OE", "AA", "QM9"]:
        start = time.time()
        append_write(conf.out_name, "start load OE \n")
        homo_lowfid = np.loadtxt(conf.json_path)
        process_time = time.time() - start
        out_time(conf.out_name, process_time)
    else:
        append_write(conf.out_name, "Dataset sould be AA or OE or QM9\n")
        append_write(conf.out_name, "program stopped ! \n")
        sys.exit()

    # -- Load for descriptor
    start = time.time()
    append_write(conf.out_name, "start load mbtr \n")
    mbtr_data = load_npz(conf.mbtr_path)
    process_time = time.time() - start
    out_time(conf.out_name, process_time)

    # -- Reduce the size of descriptor array
    if conf.mbtr_red:
        mbtr_data_red = mbtr_data[:, mbtr_data.getnnz(0) > 0]
    else:
        mbtr_data_red = mbtr_data
    # <<< End loading data

    # -- Output
    out_condition(conf.out_name, conf)
    append_write(conf.out_name, f"mbtr_data row {mbtr_data.shape[0]}")
    append_write(conf.out_name, f"mbtr_data col {mbtr_data.shape[1]}")
    mbtr_data_size = mbtr_data.shape[0]
    append_write(conf.out_name, f"mbtr_data_size {mbtr_data_size}")
    append_write(conf.out_name, f"mbtr_data_red_size {mbtr_data_red.shape[1]}")
    append_write(conf.out_name, "=============================")
    append_write(conf.out_name, f"{str(0)} -th learning")

    # check that the dataset size specified in config is correct
    assert conf.dataset_size == len(homo_lowfid), (
        f"The dataset size specified in config {conf.dataset_size}"
        " doesn't match the number of homo values"
        f"{len(homo_lowfid)}"
    )

    assert mbtr_data_red.shape[0] == len(homo_lowfid), (
        f"The number of mbtrs {mbtr_data_red.shape[0]}"
        f" doesn't match the number of homo values {len(homo_lowfid)}"
    )
    return homo_lowfid, mbtr_data_red


def plot_homo_figure(conf, homo_lowfid):
    # -- Figure of 62k dataset and AA dataset
    if conf.dataset in ["OE", "QM9", "AA"]:
        # fig_atom(df_62k,range(61489),"atoms_all.eps")
        fig_HOMO(homo_lowfid, range(conf.dataset_size), "HOMO_all.eps")
    else:
        append_write(conf.out_name, "Dataset sould be AA, OE or QM9 \n")
        append_write(conf.out_name, "program stopped ! \n")
        sys.exit()


def execute_first_training_run(conf, homo_lowfid, mbtr_data_red):
    """
    Checks if 1_full_idxs exists:
    If it does, then it prints out that the file exists and does nothing else.
    If 1_full_idxs doesn't exist, then it does the following:
    1. train_test_split based on the random seed
    2. Train a GP on the loaded data
    3. Run acquisition and get prediction, remaining and held_out set
    4. Save the data in 1_full_idxs
    5. Print, data successfully generated.
    """
    rnd = conf.random_seed

    if exists(f"{conf.out_name}_1_full_idxs.npz"):
        print("{conf.out_name}_1_full_idxs.npz exists continuing from that file")
    else:
        print(
            f"{conf.out_name}_1_full_idxs.npz Doesn't exist.\n"
            "Performing train test split."
        )
        # working with indices. Also checked that the dataset size is the
        # same as the nubmer of elements in homolow_fid and number of mbtrs
        all_indices = range(len(homo_lowfid))
        test_idxs, remaining_idxs = train_test_split(
            all_indices, train_size=conf.test_set_size, random_state=rnd
        )
        prediction_idxs, remaining_idxs = train_test_split(
            remaining_idxs, train_size=conf.pre_idxs[0], random_state=rnd
        )

        # save the indices
        idx = 0
        np.savez(
            f"{conf.out_name}_{idx}_full_idxs.npz",
            remaining_idxs=remaining_idxs,
            test_idxs=test_idxs,
            prediction_idxs=prediction_idxs,
        )

        # train GP on loaded data
        gpr = get_gp_model(
            conf, idx, prediction_idxs, test_idxs, mbtr_data_red, homo_lowfid
        )

        # run acquisition
        idx = 1
        prediction_set_size = conf.pre_idxs[idx]
        start = time.time()
        append_write(conf.out_name, "Starting acq function.\n")
        pred_idxs, rem_idxs, _, _ = acq_fn(
            conf.fn_name,
            idx,
            prediction_idxs,
            remaining_idxs,
            prediction_set_size,
            conf.rnd_size,
            mbtr_data_red,
            homo_lowfid,
            conf.K_high,
            gpr,
            conf.preprocess,
            conf.out_name,
            conf.random_seed,
        )
        append_write(conf.out_name, "Acq fun done.\n")
        process_time = time.time() - start
        out_time(conf.out_name, process_time)

        # these indices are to be used in the next iteration
        # therefore idx = idx + 1 (or idx = 1)
        print(f"Saving {idx} _full_idxs file.")
        np.savez(
            f"{conf.out_name}_{idx}_full_idxs.npz",
            remaining_idxs=rem_idxs,
            prediction_idxs=pred_idxs,
            test_idxs=test_idxs,
        )


def main(filepath):
    # >>> Load Config
    conf = Input(filepath)

    # set the random seed
    np.random.seed(conf.random_seed)

    append_write(conf.out_name, datetime.today().strftime("%Y-%m-%dT%I-%M-%S"))
    # >>> Load Dataset

    homo_low_fid, mbtr_data_red = load_data(conf)
    plot_homo_figure(conf, homo_low_fid)

    # execute the first training run
    execute_first_training_run(conf, homo_low_fid, mbtr_data_red)

    rem_idxs, pred_idxs, test_idxs, last_loaded_index = get_data_from_last_training_run(
        conf
    )
    # Now try to load the model file corresponding to idx-1
    # Since we will need that when we run the
    # acquition function for the next index.
    # idx = idx - 1
    gpr = get_gp_model(
        conf, last_loaded_index, pred_idxs, test_idxs, mbtr_data_red, homo_low_fid
    )

    # resume from next index
    idx = last_loaded_index + 1
    for idx, batch_size in enumerate(conf.pre_idxs[idx:], idx):
        append_write(
            conf.out_name,
            f"Resuming from index {idx} current batch size is {batch_size}\n",
        )
        print(idx, batch_size)
        start = time.time()

        const, length = get_gpr_params(gpr)
        append_write(conf.out_name, f"length RBF kernel before fit {length}\n")
        append_write(conf.out_name, f"constant of before fit {const} \n")

        append_write(conf.out_name, "Finished training \n")
        process_time = time.time() - start
        out_time(conf.out_name, process_time)

        # Go through acq_fn to get new pred_idxs
        prediction_set_size = batch_size
        start = time.time()
        append_write(conf.out_name, "Starting acq function.\n")
        pred_idxs, rem_idxs, x_train_pp, y_train = acq_fn(
            conf.fn_name,
            idx,
            pred_idxs,
            rem_idxs,
            prediction_set_size,
            conf.rnd_size,
            mbtr_data_red,
            homo_low_fid,
            conf.K_high,
            gpr,
            conf.preprocess,
            conf.out_name,
            conf.random_seed,
        )
        append_write(conf.out_name, "Acq fun done.\n")
        process_time = time.time() - start
        out_time(conf.out_name, process_time)
        print(f"Saving {idx} _full_idxs file.")
        np.savez(
            f"{conf.out_name}_{idx}_full_idxs.npz",
            remaining_idxs=rem_idxs,
            prediction_idxs=pred_idxs,
            test_idxs=test_idxs,
        )

        x_train_pp, x_test_pp, y_train, y_test = get_data_given_indices(
            conf, pred_idxs, test_idxs, mbtr_data_red, homo_low_fid
        )
        append_write(conf.out_name, f"Training a GP model for index {idx}")
        gpr.fit(x_train_pp, y_train)
        # save GP model
        joblib.dump(gpr, f"{conf.out_name}_{idx}_model.pkl")

        append_write(
            conf.out_name,
            f"Trained {conf.out_name}_{idx}_model.pkl and saved it to disk",
        )
        compute_stats(conf, gpr, x_test_pp, y_test)
    # <<<


if __name__ == "__main__":
    filepath = sys.argv[1]
    main(filepath)
