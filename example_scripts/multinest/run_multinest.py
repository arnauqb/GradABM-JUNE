import numpy as np
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
    cube[0] = cube[0] * 1 - 0.5


def loglike(cube, ndim, nparams):
    dates, cases, deaths, cases_by_age = get_model_prediction(
        log_beta_household=torch.tensor(cube[0]),
        log_beta_company=true_log_beta_company,
        log_beta_school=true_log_beta_school,
        log_beta_leisure=true_log_beta_leisure,
        log_beta_care_home=true_log_beta_care_home,
        log_beta_university=true_log_beta_university,
    )
    time_stamps = [-1]
    _cases = cases[time_stamps]
    _true_cases = true_cases[time_stamps]
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
true_log_beta_leisure = torch.tensor(-0.5, device=device)
true_log_beta_university = torch.tensor(-0.35, device=device)
true_log_beta_care_home = torch.tensor(-0.3, device=device)

dates, true_cases, true_deaths, true_cases_by_age = get_model_prediction(
    log_beta_household=true_log_beta_household,
    log_beta_company=true_log_beta_company,
    log_beta_school=true_log_beta_school,
    log_beta_leisure=true_log_beta_leisure,
    log_beta_university=true_log_beta_university,
    log_beta_care_home=true_log_beta_care_home,
)
#true_data = torch.distributions.Poisson(
#    true_data
#).sample() + torch.distributions.Normal(0, 1000).sample((true_data.shape[0],)).to(
#    device
#)
# true_deaths_curve = true_deaths_curve + torch.distributions.Poisson(true_deaths_curve).sample()
#true_cases_by_age = torch.distributions.Poisson(
#    true_cases_by_age
#).sample() + torch.distributions.Normal(0, 2000).sample(
#    (
#        true_cases_by_age.shape[0],
#        true_cases_by_age.shape[1],
#    )
#).to(
#    device
#)

#for i in range(5):
#   plt.plot(dates[[10, 15, 20, -1]], true_cases[[10, 15, 20, -1], i].cpu().detach().numpy(), "o-")
#plt.plot(dates[[10, 15, 20, -1]], true_cases[[10, 15, 20, -1]].cpu().detach().numpy(), "o-")
#plt.show()
#raise

cube = np.random.rand(1)
ndim = 1
nparams = 1
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
