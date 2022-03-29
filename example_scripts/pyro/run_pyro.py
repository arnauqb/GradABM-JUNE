from pathlib import Path
import torch

torch.manual_seed(0)
import numpy as np
import pyro
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt

this_path = Path(__file__).parent
import sys

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
)

from torch_june import TorchJune


device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_two_super_areas.pkl"


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        #cases = model(data, timer)["agent"].transmission.sum()
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve[[5, 10, 20, -1]]
    # return torch.diff(time_curve)
    #return time_curve


def get_model_prediction(
    log_beta_company, log_beta_household, log_beta_leisure, log_beta_school
):
    # print("----")
    # print(log_beta_company.item())
    # print(log_beta_school.item())
    # print(log_beta_household.item())
    # print(log_beta_leisure.item())
    # print("----\n")
    model = TorchJune(
        log_beta_leisure=log_beta_leisure,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_company=log_beta_company,
    )
    return run_model(model)


def get_average_prediction(
    log_beta_company, log_beta_household, log_beta_leisure, log_beta_school, n=5
):
    time_curves = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_leisure=log_beta_leisure,
        log_beta_school=log_beta_school,
    )
    for i in range(n - 1):
        time_curve = get_model_prediction(
            log_beta_company=log_beta_company,
            log_beta_household=log_beta_household,
            log_beta_leisure=log_beta_leisure,
            log_beta_school=log_beta_school,
        )
        time_curves = torch.vstack((time_curves, time_curve))
    mean_time_curve = torch.mean(time_curves, 0)
    std_time_curve = torch.std(time_curves, 0)
    return mean_time_curve, std_time_curve


def pyro_model(true_time_curve):
    log_beta_company = true_log_beta_company
    log_beta_school = true_log_beta_school
    log_beta_household = true_log_beta_household
    # log_beta_leisure = true_log_beta_leisure
    # log_beta_company = pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(0.0, 1.0)
    # ).to(device)
    # log_beta_school = pyro.sample(
    #    "log_beta_school", pyro.distributions.Uniform(0.0, 1.0)
    # ).to(device)
    # log_beta_household = pyro.sample(
    #    "log_beta_household", pyro.distributions.Uniform(0.0, 1.0)
    # ).to(device)
    log_beta_leisure = pyro.sample(
        "log_beta_leisure", pyro.distributions.Uniform(-1.0, 1.0)
    ).to(device)
    # noise = pyro.sample("noise", pyro.distributions.Uniform(0.0, 1.0))
    # prop_sigma = pyro.sample("prop_sigma", pyro.distributions.LogNormal(1.0, 1.0)).to(device)
    # print(10 ** log_beta_company.item())
    # print(10 ** log_beta_school.item())
    # print(10 ** log_beta_household.item())
    # print("----\n")
    # time_curve, std_time_curve = get_average_prediction(
    #   log_beta_company=log_beta_company,
    #   log_beta_household=log_beta_household,
    #   log_beta_leisure=log_beta_leisure,
    #   log_beta_school=log_beta_school,
    # )
    time_curve = get_model_prediction(
        log_beta_company=true_log_beta_company,
        log_beta_household=true_log_beta_household,
        log_beta_school=true_log_beta_school,
        log_beta_leisure=true_log_beta_leisure,
    )
    # print("####")
    # print(mean_time_curve)
    # print(std_time_curve)
    # print(true_time_curve)
    # print("\n")
    log_time_curve = torch.log10(time_curve)
    log_true_time_curve = torch.log10(true_time_curve)
    error = torch.log10(torch.tensor(1.0 + 0.2, device=device)) + 50 / (time_curve * (1+0.2))
    y = pyro.sample(
        "obs",
        #pyro.distributions.Poisson(time_curve),
        # pyro.distributions.Normal(time_curve, 1 / torch.sqrt(time_curve)),
        #pyro.distributions.Normal(time_curve, torch.sqrt(time_curve)),
        # CustomDist(time_curve, torch.sqrt(time_curve)),
        # pyro.distributions.Normal(time_curve, std_time_curve),
        pyro.distributions.Normal(time_curve, 0.1 * time_curve),
        #pyro.distributions.Normal(log_time_curve, 1.5*error),
        obs=true_time_curve,
        #obs=log_true_time_curve,
    )
    return y


DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

# true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
# true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
# true_log_beta_household = torch.tensor(np.log10(4.0), device=device)
# true_log_beta_leisure = torch.tensor(np.log10(1.0), device=device)
true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
true_log_beta_leisure = torch.tensor(np.log10(3.0), device=device)
true_log_beta_household = torch.tensor(np.log10(4.0), device=device)

true_data = get_model_prediction(
    log_beta_company=true_log_beta_company,
    log_beta_household=true_log_beta_household,
    log_beta_school=true_log_beta_school,
    log_beta_leisure=true_log_beta_leisure,
)

# mean_time_curve, std_time_curve = get_average_prediction(
#    log_beta_company=true_log_beta_company,
#    log_beta_household=true_log_beta_household,
#    log_beta_school=true_log_beta_school,
#    log_beta_leisure=true_log_beta_leisure,
#    n=5,
# )

sample_data = get_model_prediction(
    log_beta_company=torch.tensor(np.log10(1.4)),
    log_beta_household=torch.tensor(np.log10(1.5)),
    log_beta_school=torch.tensor(np.log10(1.3)),
    log_beta_leisure=torch.tensor(np.log10(1.6)),
)


# mean_time_curve = mean_time_curve.cpu().numpy()
# std_time_curve = std_time_curve.cpu().numpy()
# errsq = 5 * np.sqrt(mean_time_curve)

#fig, ax = plt.subplots()
#error = torch.log10(torch.tensor(1.0 + 0.2, device=device))# + 50 / (sample_data * (1+0.2))
#error = error.cpu().numpy()
#sample_data = sample_data.cpu().numpy()
#log_sample_data = np.log10(sample_data)
#ax.plot(range(len(sample_data)), sample_data)
##error = np.sqrt(sample_data)
#ax.fill_between(range(len(sample_data)), sample_data - error, sample_data+error, alpha=0.5)
##ax.fill_between(range(len(sample_data)), log_sample_data - error, log_sample_data+error, alpha=0.5)
#plt.show()
# ax.plot(mean_time_curve)
# ax.fill_between(range(len(mean_time_curve)), mean_time_curve - std_time_curve, mean_time_curve + std_time_curve, alpha=0.5)
# ax.fill_between(range(len(mean_time_curve)), mean_time_curve - errsq, mean_time_curve + errsq, alpha=0.5, color="C1")
##ax.plot(sample_data.cpu().numpy(), color = "red")
# plt.show()

temp_df = pd.DataFrame(
    columns=[
        "log_beta_company",
        "log_beta_school",
        "log_beta_household",
        "log_beta_leisure",
    ]
)


def logger(kernel, samples, stage, i, temp_df):
    if stage != "Warmup":
        for key in samples:
            unconstrained_samples = samples[key]
            constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
            temp_df.loc[i, key] = constrained_samples.cpu().item()
        temp_df.to_csv("./pyro_results.csv", index=False)


mcmc_kernel = pyro.infer.NUTS(pyro_model)
# pyro_model, step_size=1e-2, adapt_mass_matrix=False, adapt_step_size=False
# )
# mcmc_kernel = pyro.infer.HMC(pyro_model, num_steps=10, step_size=0.05)
mcmc = pyro.infer.MCMC(
    mcmc_kernel,
    num_samples=1000,
    warmup_steps=100,
    hook_fn=lambda kernel, samples, stage, i: logger(
        kernel, samples, stage, i, temp_df
    ),
)
mcmc.run(true_data)
print(mcmc.summary())
print(mcmc.diagnostics())
