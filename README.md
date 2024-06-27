## Active Learning of Molecular Data for Task-Specific Objectives

### Python Requirements
latest versions of the following packages:
* NumPy
* Pandas
* Matplotlib
* SciPy
* SKLearn

### Downloading the data
Before running the script, ensure you download the relevant dataset and save it in a directory titled `data` with one subdirectory for each `dataset`.
The datasets used in the publication can be found in the links below:
* AA : https://doi.org/10.5281/zenodo.3967308.
* OE62 : https://doi.org/10.5281/zenodo.4035923.
* QM9 : https://doi.org/10.5281/zenodo.4035918.

### Instructions 
To run the code, you need to specify the configuration in a config file (c.f. [example](https://github.com/kunalghosh/Multi_Fidelity_Prediction_GP/blob/testing_runs/mfgp/task1_new/puhti_runs/AA_A_1k/run1/input.dat)). Remember to modify the `mbtr_path` and `json_path` to the appropriate paths with the MBTR vectors and HOMO energy values of the training dataset.

Invoking `__main__.py` ([link](https://github.com/kunalghosh/Multi_Fidelity_Prediction_GP/blob/testing_runs/mfgp/task1_new/__main__.py)) with the config file will execute the specified active learning setup.
All the output data and log files are saved in the working directory.

To run multiple runs of an active learning loop, please use the cookie-cutter template defined [here](https://github.com/kunalghosh/ActiveLearning_run_dir_cookiecutter/tree/master). The template will generate the folder structure and relevant config files.
