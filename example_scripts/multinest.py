import numpy as np
import pickle
import torch
import pymultinest
from mpi4py import MPI
import numpy as np

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

from torch.distributions import Normal, LogNormal

from torch_june import TorchJune, Timer, InfectionSampler

device = f"cuda:{mpi_rank+2}" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)


def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return InfectionSampler(max_infectiousness, shape, rate, shift)


def get_data(june_data_path, device, n_seed=1):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f).to(device)
    n_agents = len(data["agent"]["id"])
    sampler = make_sampler()
    inf_params = {}
    inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = inf_params_values[0].to(device)
    inf_params["shape"] = inf_params_values[1].to(device)
    inf_params["rate"] = inf_params_values[2].to(device)
    inf_params["shift"] = inf_params_values[3].to(device)
    data["agent"].infection_parameters = inf_params
    data["agent"].transmission = torch.zeros(n_agents, device=device)
    inf_choice = np.random.choice(
        range(len(data["agent"]["id"])), n_seed, replace=False
    )
    susceptibility = np.ones(n_agents)
    is_infected = np.zeros(n_agents)
    infection_time = -1.0 * np.ones(n_agents)
    susceptibility[inf_choice] = 0.0
    is_infected[inf_choice] = 1
    infection_time[inf_choice] = 0.0
    data["agent"].susceptibility = torch.tensor(
        susceptibility, dtype=torch.float, device=device
    )
    data["agent"].is_infected = torch.tensor(
        is_infected, dtype=torch.int, device=device
    )
    data["agent"].infection_time = torch.tensor(
        infection_time, dtype=torch.float, device=device
    )
    return data


def backup_inf_data(data):
    ret = {}
    ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
    ret["is_infected"] = data["agent"].is_infected.detach().clone()
    ret["infection_time"] = data["agent"].infection_time.detach().clone()
    ret["transmission"] = data["agent"].transmission.detach().clone()
    return ret


def restore_data(data, backup):
    data["agent"].transmission = backup["transmission"].detach().clone()
    data["agent"].susceptibility = backup["susceptibility"].detach().clone()
    data["agent"].is_infected = backup["is_infected"].detach().clone()
    data["agent"].infection_time = backup["infection_time"].detach().clone()
    return data

def get_model_prediction(betas):
    b1, b2, b3, b4 = betas[0], betas[1], betas[2], betas[3]
    #print("----betas-----")
    #print(b1, b2, b3, b4)
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
        cube[i] = 10.0**(cube[i] * 3.0 - 1.0) # log-uniform between 1e-4, 1e2)

def loglike(cube, ndim, nparams):
    time_curve = get_model_prediction(cube)
    loglikelihood = torch.distributions.Normal(time_curve, torch.ones(len(time_curve), device=device)).log_prob(true_data).sum().cpu().item()
    return loglikelihood




DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
#DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = Timer(
    initial_day="2022-02-01",
    total_days=15,
    weekday_step_duration=(8, 8, 8),
    weekend_step_duration=(
        12,
        12,
    ),
    weekday_activities=(
        ("company", "school"),
        ("leisure",),
        ("household",),
    ),
    weekend_activities=(("leisure",), ("school",)),
)

true_data = get_model_prediction([2., 3., 4., 1.])

n_params = 4
output_file = "multinest"
pymultinest.run(loglike, prior, n_params, verbose = True, outputfiles_basename=output_file, resume=False)
