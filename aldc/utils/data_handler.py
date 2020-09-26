from dataclasses import dataclass
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


@dataclass
class Indices:
    training: list
    heldout: list
    test: list

class DataHandler():
    def __init__(self, dataset: Dataset, dataset_size: int, testset_size: int, batches_list: list, random_seed: int):
        super(DataHandler, self).__init__()
        self.dataset = dataset
        self.dataset_size = dataset_size
        assert self.dataset_size == len(self.dataset), f"The dataset size {self.dataset_size} and the number of elements in the dataset {len(self.dataset)} don't match."
        self.testset_size = testset_size

        self.batches_list = batches_list
        self.random_seed = random_seed
        self.iteration_indices_list = [] # dictionary which keeps a track of which iteration had what indices
        _initial_splits()

        # # following three become a small data class
        # self.heldout_set = [] # List of indices in heldout_set
        # self.training_set = [] # List of indices in training set
        # self.test_set = [] # List of indices in the test set.

    def _initial_splits(self):
        """Splits the dataset into training, heldout and test sets."""
        self.indices = np.arange(self.dataset_size)
        self.heldout_indices, self.test_indices = train_test_split(self.indices,
                                                                test_size = self.testset_size,
                                                                random_state = self.random_seed)

        training_set_size = self.batches_list[batch_index]
        self.train_indices, self.heldout_indices = train_test_split(self.heldout_indices,
                                                                    train_size = training_set_size,
                                                                    random_state = self.random_seed)
        self.iteration_indices_list.append(Indices(training=self.train_indices,
                                                   heldout=self.heldout_indices,
                                                   test=self.test_indices))
        # return self.iteration_indices_list[batch_index]

    def get_splits(self, batch_index):
        return self.iteration_indices_list[batch_index]

    def add_train_idxs(self, trainset_new_idxs):
        # get old indices
        previous_data_splits = self.iteration_indices_list[-1]
        # new training Indices
        train_indices = np.r_[previous_data_splits.train_indices, trainset_new_idxs]
        # new heldout indices
        heldout_indices = np.setdiff1d(previous_data_splits.heldout_indices, trainset_new_idxs)

        next_batch_indices = Indices(training = train_indices, heldout = heldout_indices, test = self.test_indices)
        self.iteration_indices_list.append(next_batch_indices)
