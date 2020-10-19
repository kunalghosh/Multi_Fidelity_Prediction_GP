import numpy as np
from scipy.sparse import load_npz

import torch
from torch.utils.data import Dataset


class MaterialDataset(Dataset):
    def __init__(self, feature_path : str, targets_path : str, transform = None, logger=None):
        super(MaterialDataset, self).__init__()
        self.transform = transform
        self.homo_lowfid = np.loadtxt(targets_path)
        self.features = load_npz(feature_path).toarray()

    # def get_lowfid_data(self, row):
    #     return get_level(row, level_type='HOMO', subset='PBE+vdW_vacuum')

    def __len__(self):
        return len(self.homo_lowfid)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.features[idx], self.homo_lowfid[idx])

        if self.transform:
            feature, target = sample
            sample = sle.transform(target)
            sample = (feature, target)

        return sample
