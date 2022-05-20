import numpy as np
import pandas as pd
import torch
import pymultinest
from pathlib import Path
import matplotlib.pyplot as plt
import sys, os

this_path = Path(os.path.abspath(__file__)).parent

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
    get_cases_by_age,
    get_deaths_from_symptoms,
    run_model
)

from torch_june import TorchJune
from torch_june.policies import Policies

from mpi4py import MPI
#
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

#device = f"cuda:{mpi_rank+2}"
device = f"cpu"

DATA_PATH = "/Users/arnull/code/torch_june/worlds/data_london.pkl"
#DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=2000, device=device)
n_agents = DATA["agent"]["id"].shape[0]
#people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)

true_log_beta_household = torch.tensor(-0.4, device=device)
true_log_beta_company = torch.tensor(-0.3, device=device)
true_log_beta_school = torch.tensor(-0.3, device=device)
true_log_beta_leisure = torch.tensor(-1.2, device=device)
true_log_beta_university = torch.tensor(-0.5, device=device)
true_log_beta_care_home = torch.tensor(-0.4, device=device)
time_stamps = [10, 16, 21, 29]


def prior(cube, ndim, nparams):
    cube[0] = cube[0] - 1.0
    cube[1] = cube[1] - 1.0
    cube[2] = cube[2] - 1.0
    cube[3] = cube[3] - 2.0  # leisure


def loglike(cube, ndim, nparams):
    with torch.no_grad():
        model = TorchJune(
            log_beta_household=torch.tensor(cube[0]),
            log_beta_school=torch.tensor(cube[1]),
            log_beta_company=torch.tensor(cube[2]),
            log_beta_leisure=torch.tensor(cube[3]),
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

ndim = 4
nparams = ndim

output_file = "multinest_ne_4params"
pymultinest.run(
    loglike,
    prior,
    nparams,
    verbose=True,
    outputfiles_basename=output_file,
    n_iter_before_update=1,
    resume=False,
    n_live_points=1000,
    evidence_tolerance=0.01
)
