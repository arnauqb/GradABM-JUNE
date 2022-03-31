from cProfile import Profile
from pathlib import Path
import torch
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
import numpy as np
import pyro
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt

this_path = Path(os.path.abspath(__file__)).parent
import sys

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
    group_by_symptoms,
)

from torch_june import TorchJune


device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_ne.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_two_super_areas.pkl"


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    # time_curve = torch.zeros(0, dtype=torch.float, device=device)
    time_curve = model(data, timer)["agent"].is_infected.sum()
    symptoms = group_by_symptoms(
        data["agent"].symptoms,
        model.symptoms_updater.symptoms_sampler.stages,
        device=device,
    )
    dates = [timer.date]
    while timer.date < timer.final_date:
        next(timer)
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        n_by_symptoms = group_by_symptoms(
            data["agent"].symptoms,
            model.symptoms_updater.symptoms_sampler.stages,
            device=device,
        )
        symptoms = torch.vstack((symptoms, n_by_symptoms))
        dates.append(timer.date)
    return dates, time_curve, symptoms


def get_model_prediction(**kwargs):
    print(kwargs)
    model = TorchJune(**kwargs, device=device)
    return run_model(model)


def pyro_model(true_data):
    beta_company = true_beta_company
    beta_school = true_beta_school
    beta_household = true_beta_household
    beta_university = true_beta_university
    beta_care_home = true_beta_care_home
    beta_leisure = pyro.sample(
        "beta_leisure", pyro.distributions.Uniform(0.1, 10.0)
    ).to(device)
    dates, time_curve, symptoms = get_model_prediction(
        beta_company=beta_company,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_care_home=beta_care_home,
        beta_university=beta_university,
    )
    deaths = symptoms[:,-1]
    y = pyro.sample(
        "obs",
        pyro.distributions.Normal(deaths, torch.sqrt(deaths)),
        obs=true_deaths,
    )
    return y


DATA = get_data(DATA_PATH, n_seed=100, device=device)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

true_beta_company = torch.tensor(1.0, device=device)
true_beta_school = torch.tensor(1.0, device=device)
true_beta_leisure = torch.tensor(1.0, device=device)
true_beta_household = torch.tensor(3.0, device=device)
true_beta_university = torch.tensor(1.0, device=device)
true_beta_care_home = torch.tensor(3.0, device=device)

#prof = Profile()
#prof.enable()
dates, true_data, true_symptoms = get_model_prediction(
    beta_company=true_beta_company,
    beta_household=true_beta_household,
    beta_school=true_beta_school,
    beta_leisure=true_beta_leisure,
    beta_care_home=true_beta_care_home,
    beta_university=true_beta_university,
)
true_deaths = true_symptoms[:,-1]
#prof.disable()
#prof.dump_stats("./profile.prof")

#fig, ax = plt.subplots()
#deaths = symptoms[:,-1].cpu().numpy()
#cases = true_data.cpu().numpy()
#daily_deaths = np.diff(deaths, prepend=0)
#ax.plot(dates, deaths)
##ax.plot(dates, cases)
#plt.show()

temp_df = pd.DataFrame(
    columns=[
        "beta_company",
        "beta_school",
        "beta_household",
        "beta_leisure",
        "beta_care_home",
        "beta_university",
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
mcmc.run(true_deaths)
print(mcmc.summary())
print(mcmc.diagnostics())
