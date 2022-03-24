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


def get_model_prediction(b1, b2, b3, b4):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    beta_dict = {"company": b1, "school": b2, "household": b3, "leisure": b4}
    model = TorchJune(parameters=beta_dict)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve


def pyro_model(true_time_curve):
    beta_company = pyro.sample("beta_company", pyro.distributions.Uniform(-1, 1))
    beta_school = pyro.sample("beta_school", pyro.distributions.Uniform(-1, 1))
    beta_household = pyro.sample("beta_household", pyro.distributions.Uniform(-1, 1))
    beta_leisure = pyro.sample("beta_leisure", pyro.distributions.Uniform(-1, 1))
    time_curve = get_model_prediction(
        beta_company, beta_school, beta_household, beta_leisure
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

true_data = get_model_prediction(*torch.tensor([2.0, 3.0, 4.0, 1.0], device=device))

hmc_kernel = pyro.infer.HMC(pyro_model, step_size=0.05, num_steps=10)

mcmc = pyro.infer.MCMC(
    hmc_kernel,
    num_samples=1000,
    warmup_steps=20,
)
mcmc.run(true_data)

samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
samples_df = pd.DataFrame.from_dict(samples)
samples_df.to_csv("./pyro_results.csv", index=False)
