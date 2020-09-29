import sys
import numpy as np
from aldc.utils import get_config_from_json
from aldc.utils import Metric
from aldc.dataset import MaterialDataset
from aldc.models import SKLearnGPModel

config_file_path = sys.argv[1] # the config is the only argument
config = get_config_from_json(config_file_path)

# set random seed
np.random.seed(config.random_seed)

# load dataset
# This should be generic (data should be pre-formatted, so we don't need custom loading logic)
dataset = MaterialDataset(feature_path=config.features_path, targets_path=config.targets_path)

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

metric = Metric()
# TODO : All strategies should take the same arguments so passing arguments is easier.
strategy = StrategyGetter.get_strategy(config.acquisition_name)

# active learning loop
for iter, batch_size in enumerate(data_handler): # iterates through batch_indices

    data_splits = data_handler.get_splits(iter)

    model.train(dataset[data_splits.train_indices]) # TODO : save model parameters in a list after each "train()" so we have model hyperparams

    train_predictions = model.predict(dataset[data_splits.train_indices])
    test_predictions = model.predict(dataset[data_splits.test_indices])
    # compute train and test predictions
    metric.mae("train", train_predictions,dataset[data_splits.train_indices])
    metric.mae("test", test_predictions,dataset[data_splits.test_indices])

    heldout_predictions = model.predict(dataset[data_splits.heldout_indices])
    trainset_new_idxs = strategy(data_splits.heldout_set, heldout_predictions, batch_size, random_seed = config.random_seed)

    # add the newly selected indices to the training set
    data_splits.update_splits(trainset_new_idxs)
