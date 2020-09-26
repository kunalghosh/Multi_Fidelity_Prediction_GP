import sys
import numpy as np
from aldc.utils import get_config_from_json
from aldc.dataset import AADataset
from aldc.models import SKLearnGPModel

config_file_path = sys.argv[1] # the config is the only argument
config = get_config_from_json(config_file_path)

# set random seed
np.random.seed(config.random_seed)

# load dataset
# This should be generic (data should be pre-formatted, so we don't need custom loading logic)
dataset = AADataset(feature_path=config.features_path, targets_path=config.targets_path)

# setup the model
# control from config
# The interface of SKLearnGPModel and PyTorchGP model must be same.
model = SKLearnGPModel(kernel_name = config.kernel_name,
                       random_seed = config.random_seed,
                       n_restarts  = config.n_restarts,
                       normalize_y = config.normalize_y)

data_handler = DataHandler(dataset, dataset_size = config.dataset_size,
                          testset_size = config.testset_size,
                          batches_list = config.batch_sizes,
                          random_seed = config.random_seed)

# TODO : All strategies should take the same arguments so passing arguments is easier.
strategy = StrategyGetter.get_strategy(config.acquisition_name)

# active learning loop
for iter, batch_size in enumerate(self.batches_list):

    data_splits = data_handler.get_splits(iter)

    model.train(dataset[data_splits.train_indices]) # TODO : save model parameters in a list after each "train()" so we have model hyperparams
    predictions = model.predict(dataset[data_splits.heldout_indices])

    trainset_new_idxs = strategy(data_splits.heldout_set, predictions, batch_size, random_seed = config.random_seed)

    # add the newly selected indices to the training set
    data_splits.add_train_idxs(trainset_new_idxs)
    # print MAE on the train set.
    # print MAE on the test set.
