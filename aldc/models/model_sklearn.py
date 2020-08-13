from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from aldc.models import Model

class Kernel():
    def __init__(self):
        super(Kernel, self).__init__()
        # change these later with the config values.
        self.const = 1.0 
        self.bound = 1.0
        self.length = 1.0
    
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
    def __init__(self, kernel_name: str):
        super(SKLearnModel, self).__init__()
        self.random_seed = 1234 # change it with config
        self.kernel = kernels.get(kernel_name)
        self.params = dict() # dictionary of parameters
        self.model = GaussianProcessRegressor(kernel=self.kernel,\
                random_state=self.random_seed)

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
