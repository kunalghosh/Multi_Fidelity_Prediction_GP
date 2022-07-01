from io_utils import Input
import click
import numpy as np
import glob


@click.command()
@click.argument("working_dir", default=".")
@click.option(
    "--idxs_within_energy", is_flag=True, help="Check how many indices are within energy"
)
def main(idxs_within_energy, working_dir):
    if idxs_within_energy:
        config = Input(f"{working_dir}/input.dat")
        homo_vals = np.loadtxt(config.json_path)
        range_low = float(config.range_low)
        max_num_above_range = sum(homo_vals > range_low)
        print(
            f"{config.json_path} has {max_num_above_range} values above {config.range_low} eV"
        )
        full_idxs_files = glob.glob(f"{working_dir}/*_full_idxs*")
        for file in full_idxs_files:
            idxs_ = np.load(file)["prediction_idxs"]
            num_above_range = sum(homo_vals[idxs_] > range_low)
            assert (
                num_above_range < max_num_above_range,
                f"The number above {config.range_low} cannot be above the {max_num_above_range} calculated from all the homo values",
            )
            print(
                f" For file {file} above {config.range_low} eV = {num_above_range}, % of total = {num_above_range * 100 / max_num_above_range}"
            )


if __name__ == "__main__":
    main()
