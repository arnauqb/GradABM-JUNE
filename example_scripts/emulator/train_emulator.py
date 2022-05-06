import numpy as np
import torch
import pickle
import gpytorch
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from pyDOE import lhs

this_path = Path(os.path.abspath(__file__)).parent
sys.path.append(this_path.parent.as_posix())
from script_utils import (
    get_average_predictions,
    make_timer,
    get_data,
    get_people_by_age,
    backup_inf_data,
)

device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_england.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=100, device=device)
n_agents = DATA["agent"]["id"].shape[0]
people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


log_beta_leisure = torch.tensor(-0.5, device=device)
log_beta_care_home = torch.tensor(-0.3, device=device)
def generate_samples(n_samples):
    train_x = torch.tensor(lhs(4, samples=n_samples, criterion="center"), device=device)
    train_x = train_x - 1
    train_y = torch.empty(0, device=device)
    for i in range(n_samples):
        with torch.no_grad():
            (
                dates,
                cases_mean,
                cases_std,
                deaths_mean,
                deaths_std,
                cases_by_age_mean,
                cases_by_age_std,
            ) = get_average_predictions(
                log_beta_household=train_x[i][0],
                log_beta_school=train_x[i][1],
                log_beta_company=train_x[i][2],
                log_beta_university=train_x[i][3],
                log_beta_leisure=log_beta_leisure,
                log_beta_care_home=log_beta_care_home,
                timer=TIMER,
                data=DATA,
                backup=BACKUP,
                device=device,
            )
        train_y = torch.hstack((train_y, cases_mean[-1]))
    with open("./samples.pkl", "wb") as f:
        pickle.dump((train_x, train_y), f)


n_samples = 500
#generate_samples(n_samples)
train_x, train_y = pickle.load(open("./samples.pkl", "rb"))

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = ExactGPModel(train_x, train_y, likelihood).to(device)
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter = 10000
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print(
        "Iter %d/%d - Loss: %.2e   lengthscale: %.3f   noise: %.3f"
        % (
            i + 1,
            training_iter,
            loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item(),
        )
    )
    optimizer.step()

torch.save(model.state_dict(), "emulator.pth")


# test_x = torch.linspace(-2, 2, 500).to(device)
test_x = torch.tensor([[-0.5, -0.3, -0.4, -0.35]], device=device)
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()
mean = mean.cpu()
lower = lower.cpu()
upper = upper.cpu()

with torch.no_grad():
    (
        dates,
        cases_mean,
        cases_std,
        deaths_mean,
        deaths_std,
        cases_by_age_mean,
        cases_by_age_std,
    ) = get_average_predictions(
        log_beta_household=test_x[0][0],
        log_beta_school=test_x[0][1],
        log_beta_company=test_x[0][2],
        log_beta_university=test_x[0][3],
        log_beta_leisure=log_beta_leisure,
        log_beta_care_home=log_beta_care_home,
        timer=TIMER,
        data=DATA,
        backup=BACKUP,
        device=device,
    )
# fig, ax = plt.subplots()
# ax.plot(dates, cases_mean.cpu().numpy())
# ax.plot(dates, mean)
print(f"true value {cases_mean[-1]}")
print(f"Exp. mean {mean}")
print(f"low {lower}")
print(f"up {upper}")

# train_x = train_x.cpu()
# train_y = train_y.cpu()
# test_x = test_x.cpu()
# with torch.no_grad():
#    # Initialize plot
#    f, ax = plt.subplots(1, 1, figsize=(4, 3))
#
#    # Plot training data as black stars
#    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#    # Plot predictive means as blue line
#    ax.plot(test_x.numpy(), mean.numpy(), 'b')
#    # Shade between the lower and upper confidence bounds
#    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#    #ax.set_ylim([-3, 3])
#    ax.legend(['Observed Data', 'Mean', 'Confidence'])
#
# plt.show()
# f.savefig("eval.pdf")
