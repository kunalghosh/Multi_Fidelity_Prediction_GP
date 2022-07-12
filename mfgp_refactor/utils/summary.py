from mfgp_refactor.io_utils import Input
import click
import numpy as np
import glob

default_range_low = {"AA": -8.5, "OE": -5.2, "QM9": -5.55}

def fmt(val):
    try:
        a = f"{val}"
    except Exception as e:
        a = val
    return a

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


def get_predicted_homos(working_dir, idx):
    files = glob.glob(f"{working_dir}/*{idx}_testset_predictions.npz.npy")
    try:
        predicted_vals = np.load(files[0])
    except Exception as e:
        predicted_vals = None
    return predicted_vals


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
            f"{config.json_path} has {max_num_above_range} values above {range_low} eV"
        )

        sorted_idxs_files = get_full_idxs_filenames(working_dir)

        for idx, file in enumerate(sorted_idxs_files):
            idxs_ = np.load(file)["prediction_idxs"]
            test_idxs_ = np.load(file)["test_idxs"]

            predicted_homos = get_predicted_homos(working_dir, idx)
            homo_test = homo_vals[test_idxs_]
            mae, mae_in_range = get_mae(homo_test, predicted_homos, range_low)

            num_above_range = sum(homo_vals[idxs_] > range_low)
            true_positive = get_true_positive(idx, num_above_range, len(idxs_))
            assert (
                num_above_range < max_num_above_range
            ), f"The number above {config.range_low} cannot be above the {max_num_above_range} calculated from all the homo values"

            print(
                f" For file {file} above {config.range_low} eV = {num_above_range}, % of total = {num_above_range * 100 / max_num_above_range}, test_MAE = {mae}, inRange_MAE = {mae_in_range}, true_positive = {true_positive}"
            )


if __name__ == "__main__":
    main()
