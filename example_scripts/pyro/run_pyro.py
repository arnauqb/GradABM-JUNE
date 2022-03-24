from pathlib import Path
import torch
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


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve / time_curve.max()


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
    beta_company = pyro.sample(
        "beta_company", pyro.distributions.Uniform(-0.5, 1.5)
    ).to(device)
    beta_school = pyro.sample("beta_school", pyro.distributions.Uniform(-1.5, 1.5)).to(
        device
    )
    beta_household = pyro.sample(
        "beta_household", pyro.distributions.Uniform(-1.5, 1.5)
    ).to(device)
    beta_leisure = pyro.sample(
        "beta_leisure", pyro.distributions.Uniform(-1.5, 1.5)
    ).to(device)
    time_curve = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_leisure=log_beta_leisure,
        log_beta_school=log_beta_school,
    )
    pyro.sample(
        "obs",
        pyro.distributions.Normal(
            time_curve, torch.ones(time_curve.shape[0], device=device)
        ),
        obs=true_time_curve,
    )


DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

log_beta_company = torch.tensor(0.2, device=device)
log_beta_school = torch.tensor(0.4, device=device)
log_beta_household = torch.tensor(0.5, device=device)
log_beta_leisure = torch.tensor(0.8, device=device)

true_data = get_model_prediction(
    log_beta_company=log_beta_company,
    log_beta_household=log_beta_household,
    log_beta_school=log_beta_school,
    log_beta_leisure=log_beta_leisure,
)

hmc_kernel = pyro.infer.HMC(pyro_model, step_size=0.05, num_steps=25)
nuts_kernel = pyro.infer.NUTS(pyro_model, step_size=0.01)

mcmc = pyro.infer.MCMC(
    nuts_kernel,
    num_samples=2000,
    warmup_steps=200,
)
mcmc.run(true_data)

samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
samples_df = pd.DataFrame.from_dict(samples)
samples_df.to_csv("./pyro_results.csv", index=False)
