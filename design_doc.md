s# Design Doc

## The model class

The first main component is the Model class.
It could be a _SKLearn_ model or a _PyTorch_ model or something else.

```python
class Model(object):
  """docstring for Model."""

  def __init__(self, arg):
    super(Model, self).__init__()
    self.arg = arg
    self.params = dict() # dictionary of model parameters

  def fit(self, arg):
    pass

  def predict(self, arg):
    pass

  def load_params(self, path: str):
    pass

  def save_params(self, path: str):
    pass

  def get_params(self):
    return self.params
```


For example the SKLearn GP class has the following implementation

```python
class SKLearnGPModel(Model):
  """docstring for SKLearnGPModel."""

  def __init__(self, kernel_name: str):
    super(SKLearnModel, self).__init__()
    self.kernel = kernels.get(kernel_name)
    self.params = dict() # dictionary of parameters
    self.model = GaussianProcessRegressor(kernel=kernel,\
      random_state=random_seed)

  def fit(self, X_train, Y_train):
    self.model.fit()

  def predict(self, X_test):
    self.model.predict(X_test)

  def get_params(self):
    self.params = {
                   "constant_value" : self.kernel.k1.constant_value,
                   "length_scale"   : self.kernel.k2.length_scale
                  }
    return self.params
```

## Handling the configuration file

Config file is currently a Text file it could be just a python file called `config.py`
And the config would look as follows:

```python
# config.py


#condtidion
acquisition_name        = "rnd2" # fn_name                 none
out_name                = "test" # out_name                test
dataset                 = "AA" # dataset                 AA
dataset_size            = 44004  # dataset_size            44004
testset_size            = 10000 # test_set_size           10000
rnd_size                = 2.0 # rnd_size                2.0
K_high                  = 100 # K_high                  100

#preprocess     
mbtr_read               = False # mbtr_red                False
preprocess              = none # preprocess               none

#kernel     
kernel_name             = "constRBF" # kernel_type             constRBF
length_scale_init       = 700 # length                  700
prefactor_init          = 20 # const                   20
hyperparam_bounds       = 1e2 # bound                   1e2
n_opt                   = 0 # n_opt                   0

#flag
save_load_flag          = "save" # save_load_flag          save
save_load_split_flag    = False# save_load_split_flag    False
restart_flag            = False # restart_flag            False

#active learning specific config
batch_sizes             = [
                            1000, #  for each active learning loop first batch size is the same to keep the learning comparable.
                            1000,
                            2000,
                            4000,
                            8000,
                            ]
num_total_iterations    = len(batch_sizes)

#paths
mbtr_path               = "../../../../data/AA/mbtr_k2.npz"
json_path               = "../../../../data/AA/HOMO.txt"
loadidxs_path           = "../../../../data/AA_derived/10K_dataset_load_high/10Kloadhigh2_0_idxs.npz"
loadsplitidxs_path      = "../../../../data/AA_derived/10K_dataset_load_high/10Kdataset_dataset_split_idxs.npz"
```

This would allow us to import the config as without the need for any explicit parsing.

```python
import config
config.acquisition_name # returns "rnd2"
```
### Alternative based on JSON

The further investigation the python based config file is problematic since there is no easy way to pass it to
a script as a command line parameter. Instead using a JSON based config is better.

```JSON
{
  "acquisition_name"     : "rnd2",
  "out_name"             : "test",
  "dataset"              : "AA",
  "dataset_size"         : 44004,
  "testset_size"         : 10000,
  "rnd_size"             : 2.0,
  "K_high"               : 100,
  "random_seed"          : 1234,
  "mbtr_read"            : false,
  "preprocess"           : null,
  "kernel_name"          : "constRBF",
  "length_scale_init"    : 700,
  "prefactor_init"       : 20,
  "hyperparam_bounds"    : 1e2,
  "n_opt"                : 0,
  "save_load_flag"       : "save",
  "save_load_split_flag" : false,
  "restart_flag"         : false,
  "batch_sizes"          : [
                              1000,
                              1000,
                              2000,
                              4000,
                              8000
                            ],
  "num_total_iterations" : 4,
  "mbtr_path"            : "../../../../data/AA/mbtr_k2.npz",
  "json_path"            : "../../../../data/AA/HOMO.txt",
  "loadidxs_path"        : "../../../../data/AA_derived/10K_dataset_load_high/10Kloadhigh2_0_idxs.npz",
  "loadsplitidxs_path"   : "../../../../data/AA_derived/10K_dataset_load_high/10Kdataset_dataset_split_idxs.npz"
}
```
Then the code to load the above json file would be

```python
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
    length_scale_init : int
    prefactor_init : int
    hyperparam_bounds : int
    n_opt : int

    save_load_flag : str
    save_load_split_flag : bool
    restart_flag : bool

    batch_sizes : list
    num_total_iterations : int

    mbtr_path : str
    json_path : str
    loadidxs_path : str
    loadsplitidxs_path : str

with open("config.json", "r") as json_file:
  json_data = json_file.read()

json_config = json.loads(json_data)
config = Config(**json_config)
```

## The Acqusition functions

> The following section was written by :
> * First listing down the function names.
> * Then the docstring was filled in to figure out what the function must do.
> * Subsequently the argument list (their datatypes) and the return types were identified.

```python
class StrategyA():
  def __init__(self, random_seed: int, model = None, debug = False):
    super(StrategyA, self).__init__()
    self.random_seed = random_seed
    self.model = model
    self.debug = debug
    self.strategy = self.strategy_A

  def __call__(heldout_set: list, batch_size: int) -> list:
    # if strategy needs a model, it just uses self.model internally.
    next_batch = self.strategy(heldout_set, batch_size)
    return next_batch


  def strategy_A(heldout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
    """
    Random strategy :
      1. Pick molecules randomly from the held out set.
    """
    pass

####### example strategy class above ##############



def strategy_B(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Uncertainty:
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the molecules with the highest uncertainty

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass

 def strategy_C(helout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
  """
  Clustering:
    1. Cluster the held out set, into as many clusters as the next batch_size
    2. Pick molecule closest to the cluster centers

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass

def strategy_D(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
  """
  Uncertainty and Clustering:
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Cluster the set into as many clusters as the next batch_size.
    5. Pick molecule closest to the cluster centers

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass

def strategy_E(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Combination of B. and C.
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Cluster the set into as many clusters as the next batch_size.
    5. Sort each cluster based on uncertainty and pick molecule with highest uncertainty.

  debug : boolean
    Used to enable debug logs and plots.
  """
  pass

def strategy_F(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Combination of A. and B.
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Randomly pick as many molecules as the next batch_size (NOTE !! The figure needs to be updated.)

  debug : boolean
    Used to enable debug logs and plots.
  """
  pass

def strategy_G(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Cluster and Uncertainty:
    1. Cluster the entire held out set, into as many clusters as the next batch_size.
    2. Make predictions for the entire held out set. (1 and 2 can happen independently.)
    3. Pick the molecule with the highest uncertainty in each cluster (if cluster has one element pick that).

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass
```

* Next we need one function which does clustering.

```python
def cluster(heldout_set: list, n_clusters: int) -> (list, list):
  """
  first list returned is a list of indices indicating which cluster each molecule belongs to.
  second one is the cluster centers
  """
  # return (cluster_assignment, cluster_centers)
  pass

def get_closest_to_center(heldout_set: list, cluster_assignment: list, cluster_centers: list) -> list:
  """
  Molecules in the cluster which are closest to the cluster center.
  """
  pass

```

* We also need a function which sorts a list based on criterion
```python
def sort(data: list, based_on: list) -> list:
  pass

def get_last_x(data: list, top_k: float) -> list:
  """
  if top_k = 0.5, then return second half, if top_k = 1/3 then return last 1/3 of data.
  """
  pass

def sort_and_get_last_x(data: list, based_on: list, top_k: float) -> list:
  sorted_data = sort(data, based_on)
  last_x = get_last_x(sorted_data, top_k)
  return last_x
```


## Datastructure save load class
We need a class which can load an save pre-define python lists etc.
When loading it just populates datastructures.list1 with the list so we can
do append / read etc to it.

## Timing function
```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time in function-{func.__name__} : {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer
```

The above function would be used as:

```python
@timer
def loadjson(path):
    with open(path, "r") as json_file:
        json_data = json_file.read()
    return json.loads(json_data)

loadjson("test.json") # test.json has above json config
## Elapsed time in function-loadjson : 0.0002 seconds

```
# The dataset class
One for each dataset, OE, AA etc.

```python
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

class AADataset(Dataset):
    def __init__(self, feature_path : str, targets_path : str, transform = None):
        super(AADataset, self).__init__()
        self.transform = transform
        self.homo_lowfid = np.loadtxt(targets_path)
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
```

# The data class
The objective of this class is to handle data
  * Preprocess data when its loaded
  * Given a list of indices return the X (training features) or the Y (training targets).
  * Get next batch of samples. `accepts a strategy function`.

```python
class DataLoader:
    def __init__(self, batch_sizes: list, dataset: Dataset, strategy): # strategy class (because it needs to know the GP not the data loader)
        super(DataLoader, self).__init__()
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.current_batch_index = 0 # always starts from zero.
        self.strategy = strategy

    def get_data_from_idx(self, indices: list) -> List:
        return self.dataset[indices]

    def get_next_batch(self, heldout_set: List) -> List:
        batch_size = self.batch_sizes[self.current_batch_index]
        # if strategy needs a model, it just uses self.model internally.
        self.next_batch = self.strategy(heldout_set, batch_size)
        self.current_batch_index += 1
        return self.next_batch
```

# Write the main function which does what the current main does.
```python
class ActiveLearningLoop()
  def __init__(self, config : Config):
    super(ActiveLearningLoop, self).__init__()
    self.config = config
    self.train_set = None
    self.heldout_set = None
    self.test_set = None

  def get_data_splits(self):
    # split the data into train heldout_set and test
    # we only work with indices
    pass

  def step(self):
    """take one active learning step"""
    # train model with current train set
    # send heldout_set to strategy.
    # get next set of training indices.
    # update train_set
    # update heldout_set
```

```python

# load config
# load data
# get a split of indices
# create model
# 
