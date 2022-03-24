import torch
import numpy as np
import pickle
from torch.distributions import Normal, LogNormal

from torch_june import InfectionSampler, Timer


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

def make_timer():
    return Timer(
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
