from dataclasses import dataclass
from torch.nn import L1Loss


class Metric:
    def __init__(self):
        super(Metric, self).__init__()
        self.mae =  {"train": [], "test": []} # keeps the mae metric over iterations
        # could have another metric like R_sq
        self.splits = ["train", "test"]

    def mae(self, split:str, predictions:list, targets:list):
        assert split in self.splits, f"split must be one of {self.splits}"
        loss = L1Loss()
        val = loss(predictions, targets))
        self.mae[split].append(val)
        return val
