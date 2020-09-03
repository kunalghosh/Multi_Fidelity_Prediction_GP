import torch
import gpytorch
import random
import numpy as np
from gpytorch.kernels import ScaleKernel, RBFKernel
from torch.optim.lr_scheduler import StepLR
from gpytorch.distributions import MultivariateNormal



class ExactGPModel(gpytorch.models.ExactGP):
    """docstring for ExactGPModel."""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 

    def forward(self, x):
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GPytorchGPModel():
    """docstring for GPytorchGPModel."""
    def __init__(self, x_train, y_train, likelihood):
        #super(GPytorchGPModel, self).__init__(x_train, y_train,  likelihood)
        self.random_seed = 1234 # change it with config
        self.likelihood = likelihood.double()
        self.params = dict() # dictionary of parameters
        train_x = (torch.from_numpy(x_train)).double()
        train_y = (torch.from_numpy(y_train)).double()
        self.model = ExactGPModel(train_x, train_y, self.likelihood)


    def fit(self, x_train,
              y_train,
              x_valid,
              y_valid,
              torch_optimizer=torch.optim.Adam,
              lr=0.1,
              max_epochs=500,
              lr_step=1000,
              gamma=0.5):
              
        mae_loss = torch.nn.L1Loss()
        x_train = (torch.from_numpy(x_train)).double()
        y_train = (torch.from_numpy(y_train)).double()
        x_valid = (torch.from_numpy(x_valid)).double()
        y_valid = (torch.from_numpy(y_valid)).double()

        optimizer = torch_optimizer(self.model.parameters(), lr=lr)
        self.model = self.model.double()
        self.likelihood = self.likelihood.double()
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch_optimizer(self.model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=lr_step, gamma=gamma)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        val_next = np.Inf

        for i in range(max_epochs):
            optimizer.zero_grad()
            output = self.model(x_train)
            loss = -mll(output, y_train)
            loss.backward()

            
            optimizer.step()
            scheduler.step()
            '''
            if i % 50 == 0:
                if loss < val_next:
                    val_next = loss
                else:
                    print("Early stopping.")
                    break
            '''
            if i % 50 == 0:
                #with gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=False):
                self.model.eval()
                self.likelihood.eval()
                #with torch.no_grad(), gpytorch.settings.fast_pred_var():
                with torch.no_grad(), \
                    gpytorch.settings.fast_computations( solves=False):
                    #gpytorch.settings.max_preconditioner_size(100):
                    pred = self.likelihood(self.model(x_valid.double()))
                    val_ = mae_loss(pred.mean, y_valid)
                    print("Validation error: " + str(val_.item()))
                    predt = self.likelihood(self.model(x_train))
                    mae = mae_loss(predt.mean, y_train)
                if loss < val_next:
                    val_next = loss
                else:
                    print("Early stopping.")
                    #break
                self.model.train()
                self.likelihood.train()
            
            print(
                f"Iter {i+1}/{max_epochs} - MLL {loss.item()} - train mae {mae} - connst {self.model.covar_module.outputscale.item()} - lennght {self.model.covar_module.base_kernel.lengthscale.item()}"
                # f"lengthscale {model.covar_module[1].lengthscale.item()}"
                # f" noise {model.likelihood.noise.item()}")
            )
            

    def predict(self, X_test):
        x_test = torch.from_numpy(X_test)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_computations(solves=True):
            y_pred = self.likelihood(self.model(x_test.double()))
        return y_pred.mean.numpy(), [ math.sqrt(y_pred_var) for y_pred_var in y_pred.variance.numpy()]

    def get_params(self):
        self.params = {
            "constant_value" : self.model.covar_module.outputscale.item(),
            "length_scale"   : self.model.covar_module.base_kernel.lengthscale.item()
        }
        return self.params

    def set_params(self, const, lenght_scale):
        const = torch.tensor(const).double()
        lenght_scale = torch.tensor(lenght_scale).double()
        self.model.covar_module.outputscale = const
        self.model.covar_module.base_kernel.lengthscale = lenght_scale


