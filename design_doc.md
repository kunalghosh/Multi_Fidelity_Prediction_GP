# Design Doc

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
num_total_iterations    = 4
batch_sizes             = [
                            1000,
                            1000,
                            2000,
                            4000,
                            8000,
                            ]

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
