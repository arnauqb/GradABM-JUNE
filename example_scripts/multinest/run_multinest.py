import numpy as np
import pandas as pd
import torch
import pymultinest
from pathlib import Path
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
    get_cases_by_age,
    get_deaths_from_symptoms,
)

from torch_june import TorchJune
from torch_june.policies import Policies

# from mpi4py import MPI

# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()

# device = f"cuda:{mpi_rank+1}"
device = f"cuda:0"


def run_model(model):
    # print("----")
    TIMER.reset()
    data = restore_data(DATA, BACKUP)
    data = model(data, TIMER)
    time_curve = data["agent"].is_infected.sum()
    cases_by_age = get_cases_by_age(data, device=device)
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms, device=device)

    dates = [TIMER.date]
    i = 0
    while TIMER.date < TIMER.final_date:
        i += 1
        next(TIMER)
        data = model(data, TIMER)
        cases = data["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms, device=device)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        cases_age = get_cases_by_age(data, device=device)
        cases_by_age = torch.vstack((cases_by_age, cases_age))

        dates.append(TIMER.date)
    return np.array(dates), time_curve, deaths_curve, cases_by_age


def get_model_prediction(**kwargs):
    # print(kwargs)
    # t1 = time()
    with torch.no_grad():
        model = TorchJune(**kwargs, device=device)
        ret = run_model(model)
    # t2 = time()
    # print(f"Took {t2-t1:.2f} seconds.")
    return ret


def prior(cube, ndim, nparams):
    for i in range(nparams):
        cube[i] = cube[i] * 2.0 - 1.0


def loglike(cube, ndim, nparams):
    dates, cases, deaths, cases_by_age = get_model_prediction(
        log_beta_household=torch.tensor(cube[0]),
        log_beta_company=torch.tensor(cube[1]),
        log_beta_school=torch.tensor(cube[2]),
        log_beta_university=torch.tensor(cube[3]),
        log_beta_care_home=true_log_beta_care_home,
        log_beta_leisure=true_log_beta_leisure,
    )
    time_stamps = [2, 6, 12, -1]
    #time_stamps = [-1]
    _cases = cases[time_stamps]
    _true_cases = true_cases[time_stamps]
    _deaths = deaths[time_stamps]
    _true_deaths = true_deaths[time_stamps]
    cases_by_age = cases_by_age[time_stamps, :]  # / people_by_age
    _true_cases_by_age = true_cases_by_age[time_stamps, :]  # / people_by_age
    loglikelihood = (
        torch.distributions.Normal(
            _cases,
            0.3 * _cases,
        )
        .log_prob(_true_cases)
        .sum()
        .cpu()
        .item()
    )
    return loglikelihood


# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_england.pkl"
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

TIMER = make_timer()

true_log_beta_household = torch.tensor(-0.3, device=device)
true_log_beta_company = torch.tensor(-0.4, device=device)
true_log_beta_school = torch.tensor(-0.2, device=device)
true_log_beta_university = torch.tensor(-0.35, device=device)
true_log_beta_leisure = torch.tensor(-0.5, device=device)
true_log_beta_care_home = torch.tensor(-0.3, device=device)

dates, true_cases, true_deaths, true_cases_by_age = get_model_prediction(
    log_beta_household=true_log_beta_household,
    log_beta_company=true_log_beta_company,
    log_beta_school=true_log_beta_school,
    log_beta_leisure=true_log_beta_leisure,
    log_beta_university=true_log_beta_university,
    log_beta_care_home=true_log_beta_care_home,
)


#for i in range(5):
#   plt.plot(dates[[10, 15, 20, -1]], true_cases[[10, 15, 20, -1], i].cpu().detach().numpy(), "o-")
#plt.plot(dates[[10, 15, 20, -1]], true_cases[[10, 15, 20, -1]].cpu().detach().numpy(), "o-")
n_agents = DATA["agent"].id.shape[0]

daily_cases = torch.diff(true_cases, prepend=torch.tensor([0.], device=device)).cpu().detach().numpy()
#plt.plot(dates, (true_cases / n_agents).cpu().detach().numpy(), "o-")
#plt.plot(dates, (daily_cases / n_agents), "o-")
idcs = [2, 12, -1]
plt.plot(dates[idcs], torch.log10(true_cases).cpu().detach().numpy()[idcs], "o-")
plt.plot(dates, torch.log10(true_cases).cpu().detach().numpy())
plt.show()
raise

ndim = 4
cube = np.random.rand(ndim)
nparams = ndim
ll = loglike(cube, ndim, nparams)

output_file = "multinest"
pymultinest.run(
    loglike,
    prior,
    nparams,
    verbose=True,
    outputfiles_basename=output_file,
    n_iter_before_update=1,
    resume=False,
)
