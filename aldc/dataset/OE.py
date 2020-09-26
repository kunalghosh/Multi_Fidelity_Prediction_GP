from torch.utils.data import Dataset
from scipy.sparse import load_npz

class OEDataset(Dataset):
    def __init__(self, feature_path : str, targets_path : str, transform = None):
        super(OEDataset, self).__init__()
        self.transform = transform
        self.targets_df = pd.read_json(targets_path, orient='split')
        self.num_atoms = self.targets_df["number_of_atoms"].values
        self.homo_lowfid = self.targets_df.apply(self.get_lowfid_data, axis=1).to_numpy()
        self.features = load_npz(feature_path)

    def get_lowfid_data(self, row):
        return get_level(row, level_type='HOMO', subset='PBE+vdW_vacuum')

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
