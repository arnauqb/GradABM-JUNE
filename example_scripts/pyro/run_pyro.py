from pathlib import Path
import torch
torch.manual_seed(0)
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

device = "cuda:6"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"
DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_two_super_areas.pkl"

def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve


def get_model_prediction(
    log_beta_company, log_beta_household, log_beta_leisure, log_beta_school
):
    #print("----")
    #print(log_beta_company.item())
    #print(log_beta_school.item())
    #print(log_beta_household.item())
    #print(log_beta_leisure.item())
    #print("----\n")
    model = TorchJune(
        log_beta_leisure=log_beta_leisure,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_company=log_beta_company,
    )
    return run_model(model)


def pyro_model(true_time_curve):
    #log_beta_company = true_log_beta_company
    #log_beta_school = true_log_beta_school
    #log_beta_household = true_log_beta_household
    #log_beta_leisure = true_log_beta_leisure
    log_beta_company = pyro.sample(
        "log_beta_company", pyro.distributions.Uniform(-1.0, 2.0)
    ).to(device)
    log_beta_school = pyro.sample(
        "log_beta_school", pyro.distributions.Uniform(-1.0, 2.0)
    ).to(device)
    log_beta_household = pyro.sample(
        "log_beta_household", pyro.distributions.Uniform(-1.0, 2.0)
    ).to(device)
    log_beta_leisure = pyro.sample(
        "log_beta_leisure", pyro.distributions.Uniform(-1.0, 2.0)
    ).to(device)
    print("----")
    print(log_beta_company.item())
    print(log_beta_school.item())
    print(log_beta_household.item())
    print(log_beta_leisure.item())
    print("----\n")
    n = 5
    time_curves = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_leisure=log_beta_leisure,
        log_beta_school=log_beta_school,
    )
    for i in range(n-1):
        time_curve = get_model_prediction(
            log_beta_company=log_beta_company,
            log_beta_household=log_beta_household,
            log_beta_leisure=log_beta_leisure,
            log_beta_school=log_beta_school,
        )
        time_curves = torch.vstack((time_curves, time_curve))
    mean_time_curve = torch.mean(time_curves, 0)
    std_time_curve = torch.std(time_curves, 0)
    pyro.sample(
        "obs",
        #pyro.distributions.Normal(time_curve, torch.sqrt(time_curve)),
        pyro.distributions.Normal(mean_time_curve, std_time_curve),
        #pyro.distributions.Normal(time_curve, 0.1 * torch.ones(time_curve.shape[0], device=device)),
        #pyro.distributions.Normal(time_curve, time_curve/10.0),
        obs=true_time_curve,
    )


DATA = get_data(DATA_PATH, device, n_seed=10)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

# true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
# true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
# true_log_beta_household = torch.tensor(np.log10(4.0), device=device)
# true_log_beta_leisure = torch.tensor(np.log10(1.0), device=device)
true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
true_log_beta_leisure = torch.tensor(np.log10(1.0), device=device)
true_log_beta_household = torch.tensor(np.log10(4.0), device=device)

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
#mcmc_kernel = pyro.infer.HMC(pyro_model, num_steps=25, step_size=5e-2)#, num_steps=10, step_size=0.05)
mcmc = pyro.infer.MCMC(
    mcmc_kernel,
    num_samples=200,
    warmup_steps=100,
    hook_fn=lambda kernel, samples, stage, i: logger(
        kernel, samples, stage, i, temp_df
    ),
)
mcmc.run(true_data)


