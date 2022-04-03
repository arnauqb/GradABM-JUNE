from cProfile import Profile
from torch import autograd
from pathlib import Path
from time import time
import torch


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
    fix_seed,
)

fix_seed()

# torch.autograd.set_detect_anomaly(True)

from torch_june import TorchJune


device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"


def get_deaths_from_symptoms(symptoms):
    return torch.tensor(
        symptoms["current_stage"][symptoms["current_stage"] == 6].shape[0],
        device=device,
    )


def get_cases_by_age(data):
    with torch.no_grad():
        ret = torch.zeros(5, device=device)
        ages = torch.tensor([0, 20, 40, 60, 80, 100], device=device)
        for i in range(1, len(ages)):
            mask1 = data["agent"].age < ages[i]
            mask2 = data["agent"].age > ages[i - 1]
            mask = mask1 * mask2
            ret[i-1] = ret[i-1] + data["agent"].is_infected[mask].sum()
    return ret


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    data = model(data, timer)
    time_curve = data["agent"].is_infected.sum()
    cases_by_age = get_cases_by_age(data)
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms)
    dates = [timer.date]
    while timer.date < timer.final_date:
        next(timer)
        data = model(data, timer)

        cases = data["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        cases_age = get_cases_by_age(data)
        cases_by_age = torch.vstack((cases_by_age, cases_age))

        dates.append(timer.date)
    return dates, time_curve, deaths_curve, cases_by_age


def get_model_prediction(**kwargs):
    # print(kwargs)
    # t1 = time()
    #print(kwargs["beta_household"])
    model = TorchJune(**kwargs, device=device)
    ret = run_model(model)
    # t2 = time()
    # print(f"Took {t2-t1:.2f} seconds.")
    return ret


def pyro_model(true_data):
    # beta_company = true_beta_company
    # beta_school = true_beta_school
    # beta_household = true_beta_household
    # beta_university = true_beta_university
    # beta_care_home = true_beta_care_home
    # log_beta = pyro.sample("log_beta", pyro.distributions.Uniform(-1, 1)).to(device)
    # beta = pyro.deterministic("beta", 10**log_beta)
    beta_household = 10 ** pyro.sample(
        "log_beta_household", pyro.distributions.Uniform(-1, 0)
    ).to(device)
    #beta_leisure = 10 ** pyro.sample(
    #    "log_beta_leisure", pyro.distributions.Uniform(-1, 0)
    #).to(device)
    beta_school = 10 ** pyro.sample(
        "log_beta_school", pyro.distributions.Uniform(-1, 0)
    ).to(device)
    #beta_company = 10 ** pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(-1, 0)
    #).to(device)
    beta_care_home = pyro.deterministic("beta_care_home", beta_household)
    # beta_school =  pyro.sample("beta_school", pyro.distributions.Uniform(1, 5)).to(device)
    #beta_school = pyro.deterministic("beta_school", beta_household)
    beta_university = pyro.deterministic("beta_university", beta_school)
    beta_leisure = pyro.deterministic("beta_leisure", beta_household)
    # beta_leisure =  pyro.sample("beta_leisure", pyro.distributions.Uniform(1, 5)).to(device)
    beta_company = pyro.deterministic("beta_company", beta_household)
    # beta_company =  pyro.sample("beta_company", pyro.distributions.Uniform(1, 5)).to(device)
    #sigma = pyro.sample("noise", pyro.distributions.Uniform(0, 2)).to(device)

    # print(f"beta : {beta}")
    # print(f"noise :{sigma}")
    # print("\n")

    # dates, cases, deaths, cases_by_age = get_model_prediction(
    dates, cases, deaths, cases_by_age = get_model_prediction(
        beta_company=beta_company,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_care_home=beta_care_home,
        beta_university=beta_university,
    )
    #print(cases_by_age.sum(0))
    # with pyro.plate("data", 100) as ind:
    #    print(ind)
    # cases = time_curve.sum()
    # deaths = deaths[-1]
    # true_data = true_data[-1]
    y = pyro.sample(
        "obs",
        # pyro.distributions.Normal(deaths, torch.sqrt(deaths)),
        # pyro.distributions.Normal(cases_by_age[:, ind], 0.2 * cases_by_age[:, ind]),
        #pyro.distributions.Normal(torch.log10(cases_by_age), sigma),
        #pyro.distributions.Normal(torch.log10(cases), sigma),
        #pyro.distributions.Normal(torch.log10(cases), 0.2),
        #pyro.distributions.Poisson(deaths),
        pyro.distributions.Poisson(torch.log10(cases_by_age.sum(0))),
        # obs=true_data[:, ind],
        obs=torch.log10(true_data.sum(0)),
    )
    return y


DATA = get_data(DATA_PATH, n_seed=100, device=device)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

beta_household = 0.4
beta_school = 0.6
beta_company = 0.5
beta_leisure = 0.3
true_beta_company = torch.tensor(beta_company, device=device)
true_beta_school = torch.tensor(beta_school, device=device)
true_beta_university = torch.tensor(beta_school, device=device)
true_beta_leisure = torch.tensor(beta_leisure, device=device)
true_beta_household = torch.tensor(beta_household, device=device)
true_beta_care_home = torch.tensor(beta_household, device=device)

# prof = Profile()
# prof.enable()
print("computing truth...")
# dates, true_cases, true_deaths, cases_by_age = get_model_prediction(
dates, true_cases, true_deaths, true_cases_by_age = get_model_prediction(
    beta_company=true_beta_company,
    beta_household=true_beta_household,
    beta_school=true_beta_school,
    beta_leisure=true_beta_leisure,
    beta_care_home=true_beta_care_home,
    beta_university=true_beta_university,
)
print("done")
# prof.disable()
# prof.dump_stats("./profile.prof")


def plot(dates, true_cases, true_deaths, cases_by_age):
    fig, ax = plt.subplots()
    true_cases = true_cases.cpu().numpy()
    deaths = true_deaths.cpu().numpy()
    cases_by_age = cases_by_age.cpu().numpy()
    daily_deaths = np.diff(deaths, prepend=0)
    #ax.plot(cases_by_age)
    #ax.plot(deaths)
    # ax.plot(dates, true_cases)
    cmap = plt.get_cmap("viridis")(np.linspace(0,1,5))
    labels = [20,40,60,80,100]
    for i in range(cases_by_age.shape[1]):
        ax.semilogy(dates, cases_by_age[:,i], color = cmap[i], label = labels[i])
    ax.legend()
    plt.show()


def logger(kernel, samples, stage, i, temp_df):
    if stage != "Warmup":
        for key in samples:
            unconstrained_samples = samples[key]
            constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
            temp_df.loc[i, key] = constrained_samples.cpu().item()
        temp_df.to_csv("./pyro_results.csv", index=False)


def run_mcmc():
    temp_df = pd.DataFrame()
    mcmc_kernel = pyro.infer.NUTS(pyro_model)
    # pyro_model, step_size=1e-2, adapt_mass_matrix=False, adapt_step_size=False
    # )
    # mcmc_kernel = pyro.infer.HMC(pyro_model, num_steps=10, step_size=0.05)
    mcmc = pyro.infer.MCMC(
        mcmc_kernel,
        num_samples=5000,
        warmup_steps=300,
        hook_fn=lambda kernel, samples, stage, i: logger(
            kernel, samples, stage, i, temp_df
        ),
    )
    mcmc.run(true_cases_by_age)
    print(mcmc.summary())
    print(mcmc.diagnostics())


#plot(dates, true_cases, true_deaths, true_cases_by_age)
run_mcmc()
