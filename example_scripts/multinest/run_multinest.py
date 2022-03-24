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

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_prediction(betas):
    b1, b2, b3, b4 = betas[0], betas[1], betas[2], betas[3]
    timer.reset()
    data = restore_data(DATA, BACKUP)
    beta_dict = {"company" : b1, "school" : b2, "household" : b3, "leisure" : b4}
    with torch.no_grad():
        model = TorchJune(parameters=beta_dict)
        time_curve = torch.zeros(0, dtype=torch.float).to(device)
        while timer.date < timer.final_date:
            cases = model(data, timer)["agent"].is_infected.sum()
            time_curve = torch.hstack((time_curve, cases))
            next(timer)
        return time_curve

def prior(cube, ndim, nparams):
    for i in range(4):
        cube[i] = cube[i] * 3.0 - 1.0 # log-uniform between 1e-4, 1e2)

def loglike(cube, ndim, nparams):
    time_curve = get_model_prediction(cube)
    loglikelihood = torch.distributions.Normal(time_curve, torch.ones(len(time_curve), device=device)).log_prob(true_data).sum().cpu().item()
    return loglikelihood


# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ney.pkl"
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

true_data = get_model_prediction([2., 3., 4., 1.])

n_params = 4
output_file = "multinest"
pymultinest.run(loglike, prior, n_params, verbose = True, outputfiles_basename=output_file)
