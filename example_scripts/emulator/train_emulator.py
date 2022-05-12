import numpy as np
import torch
import pickle
import gpytorch
from tqdm import tqdm
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
# DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
#DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
#TIMER = make_timer()
#DATA = get_data(DATA_PATH, n_seed=100, device=device)
#n_agents = DATA["agent"]["id"].shape[0]
#people_by_age = get_people_by_age(DATA, device)
#BACKUP = backup_inf_data(DATA)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=8
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=8, rank=1
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

samples_x, samples_y = pickle.load(open("./samples_ne.pkl", "rb"))
samples_x = samples_x.float().to(device)
samples_y = samples_y.float().to(device)
n_train = int(len(samples_x) * 0.7)
train_x = samples_x[:n_train]
train_y = samples_y[:n_train]
val_x = samples_x[n_train:]
val_y = samples_y[n_train:]

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=8).to(device)
model = MultitaskGPModel(train_x, train_y, likelihood).to(device)
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
max_training_iter = 100_000
# for i in range(training_iter):
previous_loss = None
i = 0
while True:
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    if i >= max_training_iter:
        break
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, max_training_iter, loss.item()))
    optimizer.step()
    if i % 100 == 0:
        model.eval()
        likelihood.eval()
        output_val = model(val_x)
        loss_val = -mll(output_val, val_y)
        print(f"{i}:\tVal loss: {loss_val:.2e}")
        if previous_loss is not None:
            if previous_loss <= loss_val:
                break
        previous_loss = loss_val
        model.train()
        likelihood.train()
    i += 1

torch.save(model.state_dict(), "emulator.pth")


# test_x = torch.linspace(-2, 2, 500).to(device)
#test_x = torch.tensor([[-0.5, -0.3, -0.4, -0.35]], device=device)
# Get into evaluation (predictive posterior) mode
#model.eval()
#likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
#with torch.no_grad(), gpytorch.settings.fast_pred_var():
#    observed_pred = likelihood(model(test_x))
#    mean = observed_pred.mean
#    lower, upper = observed_pred.confidence_region()
#mean = mean.cpu()
#lower = lower.cpu()
#upper = upper.cpu()
#
#with torch.no_grad():
#    (
#        dates,
#        cases_mean,
#        cases_std,
#        deaths_mean,
#        deaths_std,
#        cases_by_age_mean,
#        cases_by_age_std,
#    ) = get_average_predictions(
#        log_beta_household=test_x[0][0],
#        log_beta_school=test_x[0][1],
#        log_beta_company=test_x[0][2],
#        log_beta_university=test_x[0][3],
#        log_beta_leisure=log_beta_leisure,
#        log_beta_care_home=log_beta_care_home,
#        timer=TIMER,
#        data=DATA,
#        backup=BACKUP,
#        device=device,
#    )
## fig, ax = plt.subplots()
## ax.plot(dates, cases_mean.cpu().numpy())
## ax.plot(dates, mean)
#print(f"true value {cases_mean[-1]}")
#print(f"Exp. mean {mean}")
#print(f"low {lower}")
#print(f"up {upper}")
#
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
