import numpy as np
import torch
import pymultinest
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
)

from torch_june import TorchJune
#from mpi4py import MPI

#mpi_comm = MPI.COMM_WORLD
#mpi_rank = mpi_comm.Get_rank()

#device = f"cuda:{mpi_rank+1}"
device = f"cuda:0"


def get_deaths_from_symptoms(symptoms):
    return torch.tensor(
        symptoms["current_stage"][symptoms["current_stage"] == 7].shape[0],
        device=device,
    )

def get_cases_by_age(data):
    with torch.no_grad():
        ret = torch.zeros(5, device=device)
        ages = torch.tensor([0, 20, 40, 60, 80, 100], device=device)
        for i in range(1, len(ages)):
            mask1 = data["agent"].age < ages[i]
            mask2 = data["agent"].age > ages[i-1]
            mask = mask1 * mask2
            ret[i-1] = data["agent"].is_infected[mask].sum()
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
    print(kwargs)
    # t1 = time()
    with torch.no_grad():
        model = TorchJune(**kwargs, device=device)
        ret = run_model(model)
    # t2 = time()
    # print(f"Took {t2-t1:.2f} seconds.")
    return ret


def prior(cube, ndim, nparams):
    for i in range(4):
        cube[i] = cube[i] * 4.0 - 2.0  # log-uniform between 1e-4, 1e2)


def loglike(cube, ndim, nparams):
    dates, time_curve, deaths_curve, cases_by_age = get_model_prediction(
        beta_company=10**cube[0],
        beta_school=10**cube[1],
        beta_leisure=10**cube[2],
        beta_household=10**cube[3],
        beta_care_home=10**cube[3],
        beta_university=10**cube[1]
    )
    loglikelihood = (
        torch.distributions.Normal(
            time_curve, torch.sqrt(time_curve)#, device=device)
        )
        .log_prob(true_data)
        .sum()
        .cpu()
        .item()
    )
    return loglikelihood


#DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_england.pkl"
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

true_beta_company = torch.tensor(2.0, device=device)
true_beta_school = torch.tensor(3.0, device=device)
true_beta_leisure = torch.tensor(1.0, device=device)
true_beta_household = torch.tensor(4.0, device=device)

dates, true_data, deaths_curve, cases_by_age = get_model_prediction(
    beta_company=true_beta_company,
    beta_household=true_beta_household,
    beta_school=true_beta_school,
    beta_leisure=true_beta_leisure,
)

cube = np.random.rand(4)
ndim = 4
nparams = 4
ll = loglike(cube, ndim, nparams)

n_params = 4
output_file = "multinest"
pymultinest.run(
    loglike,
    prior,
    n_params,
    verbose=True,
    outputfiles_basename=output_file,
    n_iter_before_update=1,
    resume=False,
)
