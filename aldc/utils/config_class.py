import json

from dataclasses import dataclass

@dataclass
class Config:
    acquisition_name : str
    out_name : str
    dataset : str
    dataset_size : int
    testset_size : int
    rnd_size : int
    K_high : int
    random_seed : int

    mbtr_read : bool
    preprocess : None
    kernel_name : str
    kernel_diag_noise : float
    length_scale_init : int
    prefactor_init : int
    hyperparam_bounds : int
    n_restarts : int
    normalize_y : bool

    save_load_flag : str
    save_load_split_flag : bool
    restart_flag : bool

    batch_sizes : list
    num_total_iterations : int

    features_path : str
    targets_path : str
    loadidxs_path : str
    loadsplitidxs_path : str
