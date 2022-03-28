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


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve


def get_model_prediction(
    log_beta_company, log_beta_household, log_beta_leisure, log_beta_school
):
    model = TorchJune(
        log_beta_leisure=log_beta_leisure,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_company=log_beta_company,
    )
    return run_model(model)


def prior(cube):
    cube = cube * 4 - 2
    return cube


def loglike(cube):
    time_curve = get_model_prediction(
        log_beta_company=cube[0],
        log_beta_school=cube[1],
        log_beta_leisure=cube[2],
        log_beta_household=cube[3],
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

true_log_beta_company = torch.tensor(np.log10(2.0), device=device)
true_log_beta_school = torch.tensor(np.log10(3.0), device=device)
true_log_beta_leisure = torch.tensor(np.log10(1.0), device=device)
true_log_beta_household = torch.tensor(np.log10(4.0), device=device)

true_data = get_model_prediction(
    log_beta_company=true_log_beta_company,
    log_beta_household=true_log_beta_household,
    log_beta_school=true_log_beta_school,
    log_beta_leisure=true_log_beta_leisure,
)

dlogz = 0.5
logl_max = np.inf
# output_file = "dyne"
sampler = NestedSampler(loglike, prior, ndim=4)
# sampler.run_nested()

pbar, print_func = sampler._get_print_func(None, True)
# The main nested sampling loop.
ncall = sampler.ncall
for it, res in enumerate(sampler.sample(dlogz=dlogz)):
    if it % 100 == 0:
        with open("./dyn_results.pkl", "wb") as f:
            pickle.dump(sampler.results, f)
    ncall += res[9]
    print_func(res, sampler.it-1, ncall, dlogz=dlogz, logl_max=logl_max)

pbar.close()

sampler.add_live_points()
with open("./dyn_results.pkl", "wb") as f:
    pickle.dump(sampler.results, f)
