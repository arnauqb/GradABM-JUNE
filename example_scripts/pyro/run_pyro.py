from cProfile import Profile
from torch import autograd
from pathlib import Path
from time import time
import torch


import numpy as np
import pyro
from pyro.infer.autoguide.initialization import init_to_sample
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
    get_deaths_from_symptoms,
    get_cases_by_age,
    get_people_by_age,
    get_average_predictions,
    get_model_prediction,
)

fix_seed()

# torch.autograd.set_detect_anomaly(True)

from torch_june import TorchJune
from torch_june.cuda_utils import get_fraction_gpu_used


device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_england.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=100, device=device)
n_agents = DATA["agent"]["id"].shape[0]
people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)


def pyro_model(true_data):
    # log_beta_household = pyro.sample(
    #    "log_beta_household", pyro.distributions.Normal(-0.5, 0.5)
    # ).to(device)
    log_beta_household = pyro.sample(
        "log_beta_household", pyro.distributions.Uniform(-0.75, 0.0)
    ).to(device)
    # log_beta_school = pyro.sample(
    #    "log_beta_school", pyro.distributions.Uniform(-1.0, 0.0)
    # ).to(device)
    # log_beta_company = pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(-1.0, 0.0)
    # ).to(device)
    # log_beta_university = pyro.sample(
    #    "log_beta_university", pyro.distributions.Uniform(-1.0, 0.0)
    # ).to(device)
    # log_beta_school = pyro.deterministic("log_beta_school", log_beta_household)
    # log_beta_company = pyro.deterministic("log_beta_company", log_beta_household)
    # log_beta_university = pyro.deterministic("log_beta_university", log_beta_household)
    # log_beta_leisure = pyro.deterministic("log_beta_leisure", log_beta_household)
    # log_beta_care_home = pyro.deterministic("log_beta_care_home", log_beta_household)
    # print("---")
    # print(log_beta_household)
    # print(log_beta_school)
    # print(log_beta_company)
    # print(log_beta_university)
    # print("---\n")
    (
        true_cases_mean,
        true_cases_std,
        true_deaths_mean,
        true_deaths_std,
        true_cases_by_age_mean,
        true_cases_by_age_std,
    ) = true_data
    # model = TorchJune(
    #    log_beta_household=log_beta_household,
    #    log_beta_school=log_beta_school,
    #    log_beta_company=log_beta_company,
    #    log_beta_university=log_beta_university,
    #    log_beta_care_home=true_log_beta_care_home,
    #    log_beta_leisure=true_log_beta_leisure,
    #    device=device,
    # )
    # dates, cases, deaths, cases_by_age = run_model(model)
    (
        dates,
        cases_mean,
        cases_std,
        deaths_mean,
        deaths_std,
        cases_by_age_mean,
        cases_by_age_std,
    ) = get_average_predictions(
        log_beta_household=log_beta_household,
        log_beta_school=true_log_beta_school,
        log_beta_company=true_log_beta_company,
        log_beta_university=true_log_beta_university,
        log_beta_leisure=true_log_beta_leisure,
        log_beta_care_home=true_log_beta_care_home,
        timer=TIMER,
        data=DATA,
        backup=BACKUP,
        device=device,
    )
    time_stamps = [-1]
    cases = torch.log10(cases_mean[time_stamps])  # / n_agents
    cases_std = cases_std[time_stamps] / n_agents
    true_cases = torch.log10(true_cases_mean[time_stamps])  # / n_agents
    true_cases_std = true_cases_std[time_stamps] / n_agents
    cases_by_age = torch.log10(cases_by_age_mean[time_stamps, :])  # / people_by_age
    true_cases_by_age = torch.log10(
        true_cases_by_age_mean[time_stamps, :]
    )  # / people_by_age
    # cases_mean = pyro.sample(
    #    "cases_mean", pyro.distributions.Normal(cases, 0.2 * cases)
    # )
    error = 0.05
    #log_prob = pyro.distributions.Normal(cases, error).log_prob(true_cases)
    #print("-----------")
    #print(f"beta {log_beta_household.item()}")
    #print(f"cases {cases.item()}")
    #print(f"true cases {true_cases.item()}")
    #print(f"log_prob {log_prob.item()}")
    #print("---------\n")
    pyro.sample(
        "cases",
        pyro.distributions.Normal(cases, error),
        obs=true_cases,
    )


with torch.no_grad():
    true_log_beta_household = torch.tensor(-0.2, device=device)
    true_log_beta_company = torch.tensor(-0.4, device=device)
    true_log_beta_school = torch.tensor(-0.2, device=device)
    true_log_beta_leisure = torch.tensor(-0.5, device=device)
    true_log_beta_university = torch.tensor(-0.35, device=device)
    true_log_beta_care_home = torch.tensor(-0.3, device=device)
    (
        dates,
        true_cases_mean,
        true_cases_std,
        true_deaths_mean,
        true_deaths_std,
        true_cases_by_age_mean,
        true_cases_by_age_std,
    ) = get_average_predictions(
        log_beta_household=true_log_beta_household,
        log_beta_school=true_log_beta_school,
        log_beta_company=true_log_beta_company,
        log_beta_university=true_log_beta_university,
        log_beta_leisure=true_log_beta_leisure,
        log_beta_care_home=true_log_beta_care_home,
        timer=TIMER,
        data=DATA,
        backup=BACKUP,
        device=device,
    )


def plot(dates, **kwargs):
    fig, ax = plt.subplots()
    for key, value in kwargs.items():
        kwargs[key] = value.cpu().numpy()
    # cases_by_age = kwargs["cases_by_age"]
    true_cases_by_age = kwargs["true_cases_by_age"]
    # cases = kwargs["cases"]
    true_cases = kwargs["true_cases"]
    true_deaths = kwargs["true_deaths"]
    # deaths = kwargs["deaths"]

    # deaths = np.diff(deaths, prepend=0)
    # true_deaths = np.diff(true_deaths, prepend=0)
    time_stamps = [5, 10, 20, 25]
    true_cases_by_age = true_cases_by_age[time_stamps, :]  # / people_by_age
    for i in range(5):
        data = np.log10(true_cases_by_age[:, i])
        ax.plot(range(len(time_stamps)), data, "o-", color=f"C{i}", label=i)
        error = 0.1
        ax.fill_between(
            range(len(time_stamps)),
            data - error,
            data + error,
            color=f"C{i}",
            alpha=0.5,
        )
    # cmap = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
    # labels = [20, 40, 60, 80, 100]
    # for i in range(cases_by_age.shape[1]):
    #    ax.semilogy(dates, true_cases_by_age[:, i], color=cmap[i], label=labels[i])
    #    ax.semilogy(dates, cases_by_age[:, i], color=cmap[i], label=labels[i], linestyle="--")
    ax.legend()
    fig.savefig("./plot.pdf")
    plt.show()


def logger(kernel, samples, stage, i, dfs):
    df = dfs[stage]
    for key in samples:
        if "beta" not in key:
            continue
        unconstrained_samples = samples[key].detach()
        constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
        df.loc[i, key] = constrained_samples.cpu().item()
    df.to_csv(f"./comparison_with_multinest_{stage}.csv", index=False)


def run_mcmc():
    dfs = {"Sample": pd.DataFrame(), "Warmup": pd.DataFrame()}
    mcmc_kernel = pyro.infer.NUTS(
        pyro_model,
        #step_size=5e-1,
        #adapt_step_size=False,
    )
    # mcmc_kernel = pyro.infer.HMC(pyro_model, step_size=1e-2, num_steps=25, adapt_step_size=False)
    # pyro_model, step_size=1e-2, adapt_mass_matrix=False, adapt_step_size=False
    # )
    mcmc = pyro.infer.MCMC(
        mcmc_kernel,
        num_samples=5000,
        warmup_steps=100,
        hook_fn=lambda kernel, samples, stage, i: logger(
            kernel, samples, stage, i, dfs
        ),
    )
    mcmc.run(
        (
            true_cases_mean,
            true_cases_std,
            true_deaths_mean,
            true_deaths_std,
            true_cases_by_age_mean,
            true_cases_by_age_std,
        )
    )
    print(mcmc.summary())
    print(mcmc.diagnostics())


# plot(
#  dates,
#  #cases_by_age=cases_by_age_mean,
#  true_cases_by_age=true_cases_by_age_mean,
#  #cases=cases_mean,
#  true_cases=true_cases_mean,
#  true_deaths=true_deaths_mean,
#  #deaths=deaths_mean,
# )
run_mcmc()
