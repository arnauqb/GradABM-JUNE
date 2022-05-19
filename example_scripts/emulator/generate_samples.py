import numpy as np
import torch
import pickle
import gpytorch
from tqdm import tqdm
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
from pyDOE import lhs
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

n_samples = 240

device = f"cuda:{mpi_rank}"

this_path = Path(os.path.abspath(__file__)).parent
sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_timer,
    get_data,
    get_people_by_age,
    backup_inf_data,
    run_model,
    fix_seed,
)
from parameters import make_parameters

fix_seed(0)

from torch_june import TorchJune, Timer, Policies


# DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=2000, device=device)
PARAMETERS = make_parameters()
n_agents = DATA["agent"]["id"].shape[0]
people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)
time_stamps = [10, 30, 60, 90]


def generate_samples(n_samples):
    samples_x = torch.tensor(
        lhs(2, samples=n_samples, criterion="center"), device=device
    )
    samples_x = samples_x * 4 - 2
    #samples_x[:, 0] = samples_x[:, 0] * 4 - 2
    #samples_x[:, 1] = samples_x[:, 1] * 4 - 2
    #samples_x[:, 2] = samples_x[:, 0] * 4 - 2
    #samples_x[:, 3] = samples_x[:, 1] * 4 - 2
    #samples_x[:, 1] = samples_x[:, 1] - 0.5
    #samples_x[:, 2] = samples_x[:, 2] - 0.5
    #samples_x[:, 3] = samples_x[:, 3] - 2.0
    low = int(mpi_rank * n_samples / mpi_size)
    high = int((mpi_rank + 1) * n_samples / mpi_size)
    samples_x = samples_x[low:high, :]
    samples_y = torch.empty((0, len(time_stamps)), device=device)
    for i in tqdm(range(samples_x.shape[0])):
        with torch.no_grad():
            model = TorchJune(
                log_beta_household=samples_x[i][0],
                log_beta_school=samples_x[i][1],
                log_beta_company=true_log_beta_company, #samples_x[i][2],
                log_beta_leisure=true_log_beta_leisure, #samples_x[i][3],
                log_beta_university=true_log_beta_university,
                log_beta_care_home=true_log_beta_care_home,
                policies=Policies.from_parameters(PARAMETERS),
                device=device,
            )
            dates, cases, deaths, cases_by_age = run_model(
                model=model, timer=TIMER, data=DATA, backup=BACKUP
            )
        tosave = torch.log10(cases[time_stamps] / n_agents)
        samples_y = torch.vstack((samples_y, tosave))
    with open(f"./samples_ne.{mpi_rank:02d}.pkl", "wb") as f:
        pickle.dump((samples_x.cpu(), samples_y.cpu()), f)
    return samples_x.cpu(), samples_y.cpu()


true_log_beta_household = torch.tensor(-0.5, device=device)
true_log_beta_company = torch.tensor(0.0, device=device)
true_log_beta_school = torch.tensor(0.1, device=device)
true_log_beta_leisure = torch.tensor(-1.5, device=device)
true_log_beta_university = torch.tensor(-2.0, device=device)
true_log_beta_care_home = torch.tensor(-2.0, device=device)

samples_x, samples_y = generate_samples(n_samples)

mpi_comm.Barrier()

if mpi_rank == 0:
    fpath = f"./samples_ne.00.pkl"
    os.remove(fpath)
    for i in range(1, mpi_size):
        fpath = f"./samples_ne.{i:02d}.pkl"
        spx, spy = pickle.load(open(fpath, "rb"))
        samples_x = torch.vstack((samples_x, spx))
        samples_y = torch.vstack((samples_y, spy))
        os.remove(fpath)

    # add truth
    # samples_x = torch.vstack(
    #     (
    #         samples_x,
    #         torch.tensor(
    #             [
    #                 true_log_beta_household,
    #                 true_log_beta_school,
    #                 true_log_beta_company,
    #                 true_log_beta_leisure,
    #             ]
    #         ),
    #     )
    # )
    # with torch.no_grad():
    #     model = TorchJune(
    #         log_beta_household=true_log_beta_household,
    #         log_beta_school=true_log_beta_school,
    #         log_beta_company=true_log_beta_company,
    #         log_beta_leisure=true_log_beta_leisure,
    #         log_beta_university=true_log_beta_university,
    #         log_beta_care_home=true_log_beta_care_home,
    #         policies=Policies.from_parameters(PARAMETERS),
    #         device=device,
    #     )
    #     dates, cases, deaths, cases_by_age = run_model(
    #         model=model, timer=TIMER, data=DATA, backup=BACKUP
    #     )
    # tosave = cases[time_stamps] / n_agents
    # samples_y = torch.vstack((samples_y, tosave.to("cpu")))
    with open(f"./samples_ne_{n_samples}.pkl", "wb") as f:
        pickle.dump((samples_x, samples_y), f)
