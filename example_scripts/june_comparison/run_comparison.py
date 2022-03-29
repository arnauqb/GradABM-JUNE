import numpy as np
import torch
import pickle
import pymultinest
from pathlib import Path
import pandas as pd

this_path = Path(__file__).parent
import sys

sys.path.append(this_path.parent.as_posix())

from torch_june import TorchJune, Timer


def make_timer():
    return Timer(
        initial_day="2022-02-01",
        total_days=15,
        weekday_step_duration=(24,),
        weekend_step_duration=(24,),
        weekday_activities=(("household",),),
        weekend_activities=(("household",),),
    )

#def make_sampler():
#    max_infectiousness = LogNormal(0, 0.5)
#    shape = Normal(1.56, 0.08)
#    rate = Normal(0.53, 0.03)
#    shift = Normal(-2.12, 0.1)
#    return InfectionSampler(max_infectiousness, shape, rate, shift)

def get_data(june_data_path, device, n_seed=1):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f).to(device)
    n_agents = len(data["agent"]["id"])
    #sampler = make_sampler()
    inf_params = {}
    #inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = torch.ones(n_agents, device=device) #inf_params_values[0].to(device)
    inf_params["shape"] = torch.ones(n_agents, device=device) #inf_params_values[1].to(device)
    inf_params["rate"] = torch.ones(n_agents, device=device) #inf_params_values[2].to(device)
    inf_params["shift"] = torch.ones(n_agents, device=device) #inf_params_values[3].to(device)
    data["agent"].infection_parameters = inf_params
    data["agent"].transmission = torch.zeros(n_agents, device=device)
    inf_choice = np.random.choice(
        range(len(data["agent"]["id"])), n_seed, replace=False
    )
    susceptibility = np.ones(n_agents)
    is_infected = np.zeros(n_agents)
    infection_time = np.zeros(n_agents)
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

device = f"cuda:0"


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    time_curve = model(data, timer)["agent"].is_infected.sum()
    while timer.date < timer.final_date:
        next(timer)
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
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


DATA_PATH = "./data.pkl"

DATA = get_data(DATA_PATH, device, n_seed=1)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

log_beta_company = torch.tensor(np.log10(5.0), device=device)
log_beta_school = torch.tensor(np.log10(5.0), device=device)
log_beta_leisure = torch.tensor(np.log10(5.0), device=device)
log_beta_household = torch.tensor(np.log10(5.0), device=device)

true_data = get_model_prediction(
    log_beta_company=log_beta_company,
    log_beta_household=log_beta_household,
    log_beta_school=log_beta_school,
    log_beta_leisure=log_beta_leisure,
)

df = pd.DataFrame()
df["infected"] = true_data.cpu().numpy()
df.to_csv("results.csv", index=False)
