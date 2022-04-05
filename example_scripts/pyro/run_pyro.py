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
    get_cases_by_age
)

fix_seed()

# torch.autograd.set_detect_anomaly(True)

from torch_june import TorchJune


device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"



def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    data = model(data, timer)
    time_curve = data["agent"].is_infected.sum()
    cases_by_age = get_cases_by_age(data, device=device)
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms)
    dates = [timer.date]
    while timer.date < timer.final_date:
        next(timer)
        data = model(data, timer)

        cases = data["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        cases_age = get_cases_by_age(data, device=device)
        cases_by_age = torch.vstack((cases_by_age, cases_age))

        dates.append(timer.date)
    return dates, time_curve, deaths_curve, cases_by_age


def get_model_prediction(**kwargs):
    # print(kwargs)
    # t1 = time()
    # print(kwargs["beta_household"])
    model = TorchJune(**kwargs, device=device)
    ret = run_model(model)
    # t2 = time()
    # print(f"Took {t2-t1:.2f} seconds.")
    return ret


def pyro_model(true_data):
    # log_beta = pyro.sample("log_beta", pyro.distributions.Uniform(-1, 1)).to(device)
    # beta = pyro.deterministic("beta", 10**log_beta)
    beta_household = 10 ** pyro.sample(
        "log_beta_household", pyro.distributions.Normal(-0.5, 0.5)
    ).to(device)
    # beta_leisure = 10 ** pyro.sample(
    #    "log_beta_leisure", pyro.distributions.Uniform(-1, 0)
    # ).to(device)
    beta_school = pyro.deterministic("beta_school", beta_household)
    # beta_school = 10 ** pyro.sample(
    #    "log_beta_school", pyro.distributions.Uniform(-1, 0)
    # ).to(device)
    # beta_company = 10 ** pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(-1, 0)
    # ).to(device)
    beta_care_home = pyro.deterministic("beta_care_home", beta_household)
    # beta_school =  pyro.sample("beta_school", pyro.distributions.Uniform(1, 5)).to(device)
    # beta_school = pyro.deterministic("beta_school", beta_household)
    beta_university = pyro.deterministic("beta_university", beta_school)
    beta_leisure = pyro.deterministic("beta_leisure", beta_household)
    # beta_leisure =  pyro.sample("beta_leisure", pyro.distributions.Uniform(1, 5)).to(device)
    beta_company = pyro.deterministic("beta_company", beta_household)
    # beta_company = 10 ** pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(-1, 0)
    # ).to(device)
    # epsilon = pyro.sample("relerr", pyro.distributions.Uniform(0, 1)).to(device)
    # delta = pyro.sample("abserr", pyro.distributions.Uniform(0, 10)).to(device)
    # delta = 100
    # epsilon = 0.1

    # print(f"beta : {beta}")
    # print(f"noise :{sigma}")
    # print("\n")

    # dates, cases, deaths, cases_by_age = get_model_prediction(
    #(
    #    dates,
    #    cases_mean,
    #    cases_std,
    #    deaths_mean,
    #    deaths_std,
    #    cases_by_age_mean,
    #    cases_by_age_std,
    #) = get_true_data(
    #    beta_household=beta_household,
    #    beta_company=beta_company,
    #    beta_school=beta_school,
    #    beta_leisure=beta_leisure,
    #    n=1,
    #)
    (
        true_cases_mean,
        true_cases_std,
        true_deaths_mean,
        true_deaths_std,
        true_cases_by_age_mean,
        true_cases_by_age_std,
    ) = true_data
    # dates, cases, deaths, cases_by_age = get_model_prediction(
    #    beta_company=beta_company,
    #    beta_household=beta_household,
    #    beta_school=beta_school,
    #    beta_leisure=beta_leisure,
    #    beta_care_home=beta_care_home,
    #    beta_university=beta_university,
    # )
    dates, cases, deaths, cases_by_age = get_model_prediction(
        beta_company=beta_company,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_care_home=beta_care_home,
        beta_university=beta_university,
    )
    print("---")
    print(beta_household)
    print(deaths[-1])
    print(true_deaths_mean[-1])
    print("---")
    pyro.sample(
        "cases",
        pyro.distributions.Normal(deaths[-1], 0.5*torch.sqrt(deaths[-1]) + 10),
        obs=true_deaths_mean[-1],
    )


DATA = get_data(DATA_PATH, n_seed=100, device=device)
BACKUP = backup_inf_data(DATA)

timer = make_timer()


def get_true_data(beta_household, beta_school, beta_company, beta_leisure, n=100):

    dates, true_cases, true_deaths, true_cases_by_age = get_model_prediction(
        beta_company=beta_company,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_care_home=beta_household,
        beta_university=beta_school,
    )
    true_cases = true_cases.reshape((1, *true_cases.shape))
    true_deaths = true_deaths.reshape((1, *true_deaths.shape))
    true_cases_by_age = true_cases_by_age.reshape((1, *true_cases_by_age.shape))
    for i in range(n - 1):

        _, true_cases2, true_deaths2, true_cases_by_age2 = get_model_prediction(
            beta_company=beta_company,
            beta_household=beta_household,
            beta_school=beta_school,
            beta_leisure=beta_leisure,
            beta_care_home=beta_household,
            beta_university=beta_school,
        )
        true_cases2 = true_cases2.reshape((1, *true_cases2.shape))
        true_deaths2 = true_deaths2.reshape((1, *true_deaths2.shape))
        true_cases_by_age2 = true_cases_by_age2.reshape((1, *true_cases_by_age2.shape))
        true_cases = torch.vstack((true_cases, true_cases2))
        true_deaths = torch.vstack((true_deaths, true_deaths2))
        true_cases_by_age = torch.vstack((true_cases_by_age, true_cases_by_age2))
    true_cases_mean = true_cases.mean(0)
    true_cases_std = true_cases.std(0)
    true_deaths_mean = true_deaths.to(torch.float).mean(0)
    true_deaths_std = true_deaths.to(torch.float).std(0)
    true_cases_by_age_mean = true_cases_by_age.mean(0)
    true_cases_by_age_std = true_cases_by_age.std(0)
    return (
        dates,
        true_cases_mean,
        true_cases_std,
        true_deaths_mean,
        true_deaths_std,
        true_cases_by_age_mean,
        true_cases_by_age_std,
    )


true_beta_company = torch.tensor(0.4, device=device)
true_beta_school = torch.tensor(0.4, device=device)
true_beta_university = true_beta_school
true_beta_leisure = torch.tensor(0.4, device=device)
true_beta_household = torch.tensor(0.4, device=device)
true_beta_care_home = true_beta_household


(
    dates,
    true_cases_mean,
    true_cases_std,
    true_deaths_mean,
    true_deaths_std,
    true_cases_by_age_mean,
    true_cases_by_age_std,
) = get_true_data(
    beta_household=true_beta_household,
    beta_school=true_beta_school,
    beta_company=true_beta_company,
    beta_leisure=true_beta_leisure,
    n=1,
)
factor = 1.0
(
    dates,
    cases_mean,
    cases_std,
    deaths_mean,
    deaths_std,
    cases_by_age_mean,
    cases_by_age_std,
) = get_true_data(
    beta_household=factor * true_beta_household,
    beta_school=factor * true_beta_school,
    beta_company=factor * true_beta_company,
    beta_leisure=factor * true_beta_leisure,
    n=1,
)


def plot(dates, **kwargs):
    fig, ax = plt.subplots()
    for key, value in kwargs.items():
        kwargs[key] = value.cpu().numpy()
    cases_by_age = kwargs["cases_by_age"]
    true_cases_by_age = kwargs["true_cases_by_age"]
    cases = kwargs["cases"]
    true_cases = kwargs["true_cases"]
    true_deaths = kwargs["true_deaths"]
    deaths = kwargs["deaths"]
    print(deaths[-1])
    print(true_deaths[-1])

    #deaths = np.diff(deaths, prepend=0)
    #true_deaths = np.diff(true_deaths, prepend=0)
    ax.plot(dates, deaths, label="test", color="C0")
    ax.plot(dates, true_deaths, label="true", color="C1")
    ax.fill_between(
        dates, deaths - np.sqrt(deaths), deaths + np.sqrt(deaths), color="C0", alpha=0.5
    )
    ax.fill_between(
        dates,
        true_deaths - np.sqrt(true_deaths),
        true_deaths + np.sqrt(true_deaths),
        color="C1",
        alpha=0.5,
    )
    # cmap = plt.get_cmap("viridis")(np.linspace(0, 1, 5))
    # labels = [20, 40, 60, 80, 100]
    # for i in range(cases_by_age.shape[1]):
    #    ax.semilogy(dates, true_cases_by_age[:, i], color=cmap[i], label=labels[i])
    #    ax.semilogy(dates, cases_by_age[:, i], color=cmap[i], label=labels[i], linestyle="--")
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
    mcmc_kernel = pyro.infer.NUTS(pyro_model, init_strategy=init_to_sample)
    # pyro_model, step_size=1e-2, adapt_mass_matrix=False, adapt_step_size=False
    # )
    mcmc = pyro.infer.MCMC(
        mcmc_kernel,
        num_samples=5000,
        warmup_steps=10,
        hook_fn=lambda kernel, samples, stage, i: logger(
            kernel, samples, stage, i, temp_df
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


#plot(
#    dates,
#    cases_by_age=cases_by_age_mean,
#    true_cases_by_age=true_cases_by_age_mean,
#    cases=cases_mean,
#    true_cases=true_cases_mean,
#    true_deaths=true_deaths_mean,
#    deaths=deaths_mean,
#)
run_mcmc()
