import numpy as np
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
)

from torch_june import TorchJune

# from mpi4py import MPI

# mpi_comm = MPI.COMM_WORLD
# mpi_rank = mpi_comm.Get_rank()

# device = f"cuda:{mpi_rank+2}"
device = f"cuda:0"


def get_deaths_from_symptoms(symptoms):
    return torch.tensor(
        symptoms["current_stage"][symptoms["current_stage"] == 7].shape[0],
        device=device,
    )


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = model(data, timer)["agent"].is_infected.sum()
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms)
    dates = [timer.date]
    while timer.date < timer.final_date:
        next(timer)
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        dates.append(timer.date)
    return dates, time_curve, deaths_curve


def get_model_prediction(**kwargs):
    #print(kwargs)
    # t1 = time()
    model = TorchJune(**kwargs, device=device)
    ret = run_model(model)
    # t2 = time()
    # print(f"Took {t2-t1:.2f} seconds.")
    return ret


def prior(cube):
    cube = cube * 4 - 2
    return cube

def loglike(cube):
    cube = 10 ** cube
    _, time_curve, _ = get_model_prediction(
        beta_company=cube[0],
        beta_school=cube[1],
        beta_leisure=cube[2],
        beta_household=cube[3],
        beta_university =cube[4],
        beta_care_home =cube[5],
    )
    loglikelihood = (
        torch.distributions.Normal(
            time_curve, torch.sqrt(time_curve)  # , device=device)
        )
        .log_prob(true_data)
        .sum()
        .cpu()
        .item()
    )
    return loglikelihood


# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

true_beta_company = torch.tensor(1.0, device=device)
true_beta_school = torch.tensor(1.0, device=device)
true_beta_leisure = torch.tensor(1.0, device=device)
true_beta_household = torch.tensor(3.0, device=device)
true_beta_university = torch.tensor(1.0, device=device)
true_beta_care_home = torch.tensor(3.0, device=device)

dates, true_data, true_deaths = get_model_prediction(
    beta_company=true_beta_company,
    beta_household=true_beta_household,
    beta_school=true_beta_school,
    beta_leisure=true_beta_leisure,
    beta_care_home=true_beta_care_home,
    beta_university=true_beta_university,
)

dlogz = 0.5
logl_max = np.inf
# output_file = "dyne"
sampler = NestedSampler(loglike, prior, ndim=6)
# sampler.run_nested()

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
