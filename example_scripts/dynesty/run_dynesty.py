import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
from dynesty import NestedSampler
from pathlib import Path

this_path = Path(__file__).parent
import sys

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
    run_model
)

from torch_june import TorchJune

# from mpi4py import MPI

# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()

# device = f"cuda:{mpi_rank+2}"
device = f"cuda:5"

DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=2000, device=device)
n_agents = DATA["agent"]["id"].shape[0]
# people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)

true_log_beta_household = torch.tensor(-0.4, device=device)
true_log_beta_company = torch.tensor(-0.3, device=device)
true_log_beta_school = torch.tensor(-0.3, device=device)
true_log_beta_leisure = torch.tensor(-1.2, device=device)
true_log_beta_university = torch.tensor(-0.5, device=device)
true_log_beta_care_home = torch.tensor(-0.4, device=device)
time_stamps = [10, 16, 21, 29]


def prior(u):
    u[:3] = u[:3] - 1.0
    u[3] = u[3] - 2.0
    return u


def loglike(x):
    with torch.no_grad():
        model = TorchJune(
            log_beta_household=torch.tensor(x[0]),
            log_beta_school=torch.tensor(x[1]),
            log_beta_company=torch.tensor(x[2]),
            log_beta_leisure=torch.tensor(x[3]),
            log_beta_university=true_log_beta_university,
            log_beta_care_home=true_log_beta_care_home,
            device=device,
        )
        dates, cases, deaths, cases_by_age = run_model(
            model=model, timer=TIMER, data=DATA, backup=BACKUP
        )
    _cases = cases[time_stamps] / n_agents
    _true_cases = true_cases[time_stamps] / n_agents
    # _deaths = deaths[time_stamps]
    # _true_deaths = true_deaths[time_stamps]
    # cases_by_age = cases_by_age[time_stamps, :]  # / people_by_age
    # _true_cases_by_age = true_cases_by_age[time_stamps, :]  # / people_by_age
    loglikelihood = (
        torch.distributions.Normal(
            _cases,
            0.05,
        )
        .log_prob(_true_cases)
        .sum()
        .cpu()
        .item()
    )
    return loglikelihood

with torch.no_grad():
    model = TorchJune(
        log_beta_household=true_log_beta_household,
        log_beta_school=true_log_beta_school,
        log_beta_company=true_log_beta_company,
        log_beta_leisure=true_log_beta_leisure,
        log_beta_university=true_log_beta_university,
        log_beta_care_home=true_log_beta_care_home,
        device=device,
    )
    dates, true_cases, true_deaths, true_cases_by_age = run_model(
        model=model, timer=TIMER, data=DATA, backup=BACKUP
    )

dlogz = 0.5
logl_max = np.inf
sampler = NestedSampler(loglike, prior, ndim=4)

pbar, print_func = sampler._get_print_func(None, True)
# The main nested sampling loop.
ncall = sampler.ncall
for it, res in enumerate(sampler.sample(dlogz=dlogz)):
    if it % 100 == 0:
        with open("./dyn_results.pkl", "wb") as f:
            pickle.dump(sampler.results, f)
    ncall += res[9]
    print_func(res, sampler.it - 1, ncall, dlogz=dlogz, logl_max=logl_max)

pbar.close()

sampler.add_live_points()
with open("./dyn_results.pkl", "wb") as f:
    pickle.dump(sampler.results, f)
