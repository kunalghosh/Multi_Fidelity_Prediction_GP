from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from aldc.models import Model

class Kernel():
    def __init__(self):
        super(Kernel, self).__init__()
        # change these later with the config values.
        self.const = 20.0
        self.bound = 100.0
        self.length = 700.0

    def get(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise RuntimeError("Kernel {name} not found")

    def constRBF(self):
        const, bound, length = self.const, self.bound, self.length
        kernel = ConstantKernel(constant_value = const , constant_value_bounds=(const*1.0/bound, const*bound)) \
                 * RBF(length_scale=length, length_scale_bounds=(length*1.0/bound, length*bound)) # best result.
        return kernel

class SKLearnGPModel(Model):
    """docstring for SKLearnGPModel."""
    def __init__(self, kernel_name: str, n_restarts: int, random_seed: int, normalize_y: bool, logger=None):
        super(SKLearnGPModel, self).__init__()
        self.random_seed = random_seed # change it with config
        kernels = Kernel()
        self.kernel = kernels.get(kernel_name)()
        self.params = dict() # dictionary of parameters
        self.n_restarts = n_restarts
        self.normalize_y = normalize_y
        self.model = GaussianProcessRegressor(kernel = self.kernel,\
                random_state = self.random_seed,
                n_restarts_optimizer   = self.n_restarts,
                normalize_y  = self.normalize_y)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        #print(self.model.kernel_.get_params())

    def predict(self, X_test):
        mu, std = self.model.predict(X_test, return_std=True)
        return mu, std

    def get_params(self):
        self.params = {
            "constant_value" : self.model.kernel_.get_params()['k1__constant_value'],
            "length_scale"   : self.model.kernel_.get_params()['k2__length_scale']
        }
        return self.params

    def set_params(self, const, lenght_scale):
        self.model.kernel.k1.constant_value = const
        self.model.kernel.k2.length_scale = lenght_scale
