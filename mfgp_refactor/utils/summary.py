from mfgp_refactor.io_utils import Input
from collections import namedtuple

import click
import numpy as np
import glob

ClassificationScore = namedtuple("ClassificationScore", "tp fp tn fn tpr fpr".split())
default_range_low = {"AA": -8.5, "OE": -5.2, "QM9": -5.55}


def fmt(val):
    try:
        if isinstance(val, np.int64):
            a = f"{val:>6d}"
        else:
            a = f"{val:>9.3f}"
    except Exception as e:
        a = val
    return a


def get_in_range(vector_of_values, range_low):
    """
    Arguments:
        vector_of_values: numpy array of floats
        range_low: float value representing lower limit of the range.

    Returns a boolean vector:
        with True for entries above range_low, and False otherwise.
    """
    return vector_of_values > range_low


def get_true_positive_false_negative(homo_vals, predicted_vals, range_low):
    """
    Given an array of true and predicted values, and range_low (anything above range_low is correct classification). Returns entries of the confusion matrix.

    Parameters
    ----------
    homo_vals: ndarray
        1D array of `float`
    predicted_vals: ndarray
        1D array of `float`
    range_low: float
        lower limit of the range, anything above this is classified to be true

    Returns
    -------
    tuple with four `floats`. Values corresponding to entries of a confusion matrix
    """
    true = get_in_range(predicted_vals, range_low)
    positive = get_in_range(homo_vals, range_low)

    false = np.invert(true)
    negative = np.invert(positive)

    true_positive = np.sum(np.logical_and(true, positive))
    true_negative = np.sum(np.logical_and(true, negative))
    false_positive = np.sum(np.logical_and(false, positive))
    false_negative = np.sum(np.logical_and(false, negative))

    return true_positive, true_negative, false_positive, false_negative


def get_true_positive(idx, num_in_range, predicted_in_range, percent=True):
    """
    How many were `in range` out of the total `predicted to be in range`
    """
    true_positive = None
    if idx > 0:
        # because iteration 0 is randomly picked
        true_positive = num_in_range / predicted_in_range
        if percent:
            true_positive = true_positive * 100
    return true_positive


def get_full_idxs_filenames(working_dir):
    full_idxs_files = glob.glob(f"{working_dir}/*_full_idxs*")
    sorted_idxs_files = sorted(full_idxs_files, key=lambda x: int(x.split("_")[-3]))
    return sorted_idxs_files


def get_testset_predicted_homos(working_dir, idx):
    files = glob.glob(f"{working_dir}/*{idx}_testset_predictions.npz.npy")
    try:
        predicted_vals = np.load(files[0])
    except Exception as e:
        print(
            f"WARNING: Couldn't find any _testset_predictions.npz.npy file , did you run the get_mae script ? predicted_values set to None"
        )
        predicted_vals = None
    return predicted_vals


def get_heldoutset_predicted_homos(working_dir, idx):
    # idx of heldout set is in debug_mean_pred_{idx+1} if the testset_predictions is for idx
    files = glob.glob(f"{working_dir}/*_debug_mean_pred_{idx+1}_idxs.npz.npy")
    try:
        heldout_vals = np.load(files[0])
    except Exception as e:
        print(
            f"WARNING: Couldn't find any _debug_mean_pred_ file, did you run the get_mae script ? predicted_values set to None"
        )
        heldout_vals = None
    return heldout_vals


def get_mae(test_homos, predicted_homos, range_low):
    if predicted_homos is None:
        mae, mae_in_range = None, None
    else:
        error_overall = np.abs(test_homos - predicted_homos)
        mask_test_homos_in_range = test_homos > range_low  # mask of test_homos in range
        # print(test_homos[mask_test_homos_in_range])
        # print(error_overall)
        # print(mask_test_homos_in_range)
        error_in_range = error_overall[mask_test_homos_in_range]
        # print(error_in_range)
        mae, mae_in_range = np.mean(error_overall), np.mean(error_in_range)
    return mae, mae_in_range


def get_classification_metrics(true_homos, predicted_homos, range_low):
    try:
        tp, tn, fp, fn = get_true_positive_false_negative(
            true_homos, predicted_homos, range_low
        )
        tpr = tp / (tp + fn)  # Precision
        fpr = fp / (fp + tn)  # Recall
    except Exception as e:
        print(f"Couldn't compute entries of confusion matrix: {e}")
        tp, fp, tn, fn, tpr, fpr = None, None, None, None, None, None
    score = ClassificationScore(tp, fp, tn, fn, tpr, fpr)
    return score


@click.command()
@click.argument("working_dir", default=".")
@click.option(
    "--idxs_within_energy",
    is_flag=True,
    help="Check how many indices are within energy",
)
def main(idxs_within_energy, working_dir):
    if idxs_within_energy:
        config = Input(f"{working_dir}/input.dat")
        homo_vals = np.loadtxt(config.json_path)

        range_low = config.range_low
        if range_low is None:
            print(
                "WARNING : Can't find range_low value in config. Using default range_low values."
            )
            range_low = default_range_low[config.dataset]
        range_low = float(range_low)

        max_num_above_range = sum(homo_vals > range_low)
        print(
            f"{config.json_path} has {max_num_above_range} values above {fmt(range_low)} eV"
        )

        sorted_idxs_files = get_full_idxs_filenames(working_dir)

        for idx, file in enumerate(sorted_idxs_files):
            idxs_ = np.load(file)["prediction_idxs"]
            test_idxs_ = np.load(file)["test_idxs"]
            held_idxs_ = np.load(file)["remaining_idxs"]

            # Testset classification score
            testset_predicted_homos = get_testset_predicted_homos(working_dir, idx)
            if testset_predicted_homos is None:
                mae, mae_in_range = None, None
            else:
                homo_test = homo_vals[test_idxs_]
                mae, mae_in_range = get_mae(homo_test, testset_predicted_homos, range_low)

            num_above_range = sum(homo_vals[idxs_] > range_low)
            assert (
                num_above_range < max_num_above_range
            ), f"The number above {config.range_low} cannot be above the {max_num_above_range} calculated from all the homo values"
            # true_positive = get_true_positive(idx, num_above_range, len(idxs_))

            testset_score_ = get_classification_metrics(
                homo_vals[test_idxs_], testset_predicted_homos, range_low
            )

            try:
                tp, tn, fp, fn = get_true_positive_false_negative(
                    homo_vals[test_idxs_], testset_predicted_homos, range_low
                )
                tpr = tp / (tp + fn)  # Precision
                fpr = fp / (fp + tn)  # Recall
            except Exception as e:
                print(f"Couldn't compute entries of confusion matrix: {e}")
                tp, fp, tn, fn, tpr, fpr = None, None, None, None, None, None
            testset_score = ClassificationScore(tp, fp, tn, fn, tpr, fpr)
            assert testset_score_ == testset_score, "Testset scores must match"

            # ----------------------------------------------------------------------
            # heldoutset classification score
            # -----------------------------------------------------------------------
            heldout_predicted_homos = get_heldoutset_predicted_homos(working_dir, idx)
            # if heldout_predicted_homos is None:
            #     mae, mae_in_range = None, None
            # else:
            #     homo_heldout = homo_vals[held_idxs_]
            #     # mae, mae_in_range = get_mae(
            #     #     homo_heldout, heldout_predicted_homos, range_low
            #     # )
            heldout_score_ = get_classification_metrics(
                homo_vals[held_idxs_], heldout_predicted_homos, range_low
            )
            try:
                tp, tn, fp, fn = get_true_positive_false_negative(
                    homo_vals[held_idxs_], heldout_predicted_homos, range_low
                )
                tpr = tp / (tp + fn)  # Precision
                fpr = fp / (fp + tn)  # Recall
            except Exception as e:
                print(f"Couldn't compute entries of confusion matrix: {e}")
                tp, fp, tn, fn, tpr, fpr = None, None, None, None, None, None
            heldout_score = ClassificationScore(tp, fp, tn, fn, tpr, fpr)
            assert heldout_score_ == heldout_score, "Heldout scores must match"

            print(
                f"For file {file} above {fmt(config.range_low)} eV = {fmt(num_above_range)}, % of total = {fmt(num_above_range * 100 / max_num_above_range)}, test_MAE = {fmt(mae)}, inRange_MAE = {fmt(mae_in_range)}, \n"
                + "Testset\n"
                + f"tp = {fmt(testset_score.tp)}, fp = {fmt(testset_score.fp)}, tn = {fmt(testset_score.tn)}, fn = {fmt(testset_score.fn)}, tpr = {fmt(testset_score.tpr)}, fpr = {fmt(testset_score.fpr)}\n"
                + "Heldoutset\n"
                + f"tp = {fmt(heldout_score.tp)}, fp = {fmt(heldout_score.fp)}, tn = {fmt(heldout_score.tn)}, fn = {fmt(heldout_score.fn)}, tpr = {fmt(heldout_score.tpr)}, fpr = {fmt(heldout_score.fpr)}\n"
            )


if __name__ == "__main__":
    main()
