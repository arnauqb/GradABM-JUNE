from pathlib import Path
import torch
import numpy as np
import pyro
import pandas as pd
import json

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

gaussian_kernel = torch.autograd.Variable(
    torch.tensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]], device=device)
)


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    last_cases = 0.0
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        daily_cases = cases - last_cases
        last_cases = daily_cases.item()
        time_curve = torch.hstack((time_curve, daily_cases))
        next(timer)
    time_curve = time_curve.reshape((1, time_curve.shape[0]))
    time_curve = torch.nn.functional.conv1d(time_curve, gaussian_kernel)
    return time_curve.squeeze() / data["agent"].id.shape[0]


def get_model_prediction(
    log_beta_company, log_beta_household, log_beta_leisure, log_beta_school
):
    model = TorchJune(
        log_beta_leisure=log_beta_leisure,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_company=log_beta_company,
    )
    return run_model(model)


def pyro_model(true_time_curve):
    log_beta_company = pyro.sample(
        "log_beta_company", pyro.distributions.Uniform(0.0, 1.0)
    ).to(device)
    log_beta_school = pyro.sample(
        "log_beta_school", pyro.distributions.Uniform(0.0, 1.0)
    ).to(device)
    log_beta_household = pyro.sample(
        "log_beta_household", pyro.distributions.Uniform(0.0, 1.0)
    ).to(device)
    log_beta_leisure = pyro.sample(
        "log_beta_leisure", pyro.distributions.Uniform(0.0, 1.0)
    ).to(device)
    #print("----")
    #print(log_beta_company.item())
    #print(log_beta_school.item())
    #print(log_beta_household.item())
    #print(log_beta_leisure.item())
    #print("----")
    time_curve = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_leisure=log_beta_leisure,
        log_beta_school=log_beta_school,
    )
    pyro.sample(
        "obs",
        pyro.distributions.Normal(
            time_curve, 3 * torch.ones(time_curve.shape[0], device=device)
        ),
        obs=true_time_curve,
    )


DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

# true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
# true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
# true_log_beta_household = torch.tensor(np.log10(4.0), device=device)
# true_log_beta_leisure = torch.tensor(np.log10(1.0), device=device)
true_log_beta_company = torch.tensor(np.log10(3.0), device=device)
true_log_beta_school = torch.tensor(np.log10(4.0), device=device)
true_log_beta_household = torch.tensor(np.log10(2.0), device=device)
true_log_beta_leisure = torch.tensor(np.log10(6.0), device=device)

true_data = get_model_prediction(
    log_beta_company=true_log_beta_company,
    log_beta_household=true_log_beta_household,
    log_beta_school=true_log_beta_school,
    log_beta_leisure=true_log_beta_leisure,
)

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
#mcmc_kernel = pyro.infer.HMC(pyro_model)
mcmc = pyro.infer.MCMC(
    mcmc_kernel,
    num_samples=10000,
    warmup_steps=1000,
    hook_fn=lambda kernel, samples, stage, i: logger(
        kernel, samples, stage, i, temp_df
    ),
)
mcmc.run(true_data)
