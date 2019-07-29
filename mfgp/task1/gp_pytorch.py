import pdb
import gpytorch
import torch
from torch.optim.lr_scheduler import StepLR
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel, MaternKernel
from sklearn.cluster import KMeans
from gpytorch.distributions import MultivariateNormal


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, z):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.M = 500  # number of inducing points
        if z is None:
            z = train_x[:self.M]
        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=z,
                                                likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactGPModel_Matern(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, z):
        super(ExactGPModel_Matern, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = ScaleKernel(MaternKernel(nu=2.5))
        self.M = 500  # number of inducing points
        if z is None:
            z = train_x[:self.M]
        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=z,
                                                likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def model_fit(model,
              likelihood,
              x_train,
              y_train,
              torch_optimizer=torch.optim.Adam,
              lr=0.1,
              max_epochs=50):

    model.train()
    likelihood.train()

    optimizer = torch_optimizer(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=250, gamma=0.5)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(max_epochs):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        print(
            f"Iter {i+1}/{max_epochs} - Loss {loss.item()} "
            # f"lengthscale {model.covar_module[1].lengthscale.item()}"
            # f" noise {model.likelihood.noise.item()}")
        )
        scheduler.step()
        if isinstance(optimizer, torch.optim.LBFGS):
            closure = lambda: -mll(model(x_train), y_train)
            optimizer.step(closure)
        else:
            optimizer.step()


def model_predict(model=None, likelihood=None, x_test=None):
    assert model is not None, f"Model is {model}, pass a valid GPyTorch model"
    assert likelihood is not None, f"Likelihood is {likelihood}, pass same GPyTorch likelihood used during training."
    assert x_test is not None, "x_test is {x_test}, pass test points at which to make predictions."

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
        return observed_pred
