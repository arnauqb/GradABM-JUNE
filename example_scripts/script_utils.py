import torch
import numpy as np
import pickle
from torch.distributions import Normal, LogNormal

from torch_june import TransmissionSampler, Timer


def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return TransmissionSampler(max_infectiousness, shape, rate, shift)


def infector(data, indices, device):
    susc = data["agent"]["susceptibility"].cpu().numpy()
    is_inf = data["agent"]["is_infected"].cpu().numpy()
    inf_t = data["agent"]["infection_time"].cpu().numpy()
    next_stage = data["agent"]["symptoms"]["next_stage"].cpu().numpy()
    susc[indices] = 0.0
    is_inf[indices] = 1.0
    inf_t[indices] = 0.0
    next_stage[indices] = 2
    data["agent"]["susceptibility"] = torch.tensor(susc, device=device)
    data["agent"]["is_infected"] = torch.tensor(is_inf, device=device)
    data["agent"]["infection_time"] = torch.tensor(inf_t, device=device)
    data["agent"]["symptoms"]["next_stage"] = torch.tensor(next_stage, device=device)
    return data


def get_data(june_data_path, device, n_seed=10):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f).to(device)
    n_agents = len(data["agent"]["id"])
    # sampler = make_sampler()
    inf_params = {}
    # inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[0].to(device)
    inf_params["shape"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[1].to(device)
    inf_params["rate"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[2].to(device)
    inf_params["shift"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[3].to(device)
    data["agent"].infection_parameters = inf_params
    data["agent"].transmission = torch.zeros(n_agents, device=device)
    data["agent"].susceptibility = torch.ones(n_agents, device=device)
    data["agent"].is_infected = torch.zeros(n_agents, device=device)
    data["agent"].infection_time = torch.zeros(n_agents, device=device)
    symptoms = {}
    symptoms["current_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
    symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
    symptoms["time_to_next_stage"] = torch.zeros(n_agents, device=device)
    data["agent"].symptoms = symptoms
    indices = np.arange(0, n_agents)
    np.random.shuffle(indices)
    indices = indices[:n_seed]
    data = infector(data=data, indices=indices, device=device)
    return data


def backup_inf_data(data):
    ret = {}
    ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
    ret["is_infected"] = data["agent"].is_infected.detach().clone()
    ret["infection_time"] = data["agent"].infection_time.detach().clone()
    ret["transmission"] = data["agent"].transmission.detach().clone()
    symptoms = {}
    symptoms["current_stage"] = (
        data["agent"]["symptoms"]["current_stage"].detach().clone()
    )
    symptoms["next_stage"] = data["agent"]["symptoms"]["next_stage"].detach().clone()
    symptoms["time_to_next_stage"] = (
        data["agent"]["symptoms"]["time_to_next_stage"].detach().clone()
    )
    ret["symptoms"] = symptoms
    return ret


def restore_data(data, backup):
    data["agent"].transmission = backup["transmission"].detach().clone()
    data["agent"].susceptibility = backup["susceptibility"].detach().clone()
    data["agent"].is_infected = backup["is_infected"].detach().clone()
    data["agent"].infection_time = backup["infection_time"].detach().clone()
    data["agent"].symptoms["current_stage"] = (
        backup["symptoms"]["current_stage"].detach().clone()
    )
    data["agent"].symptoms["next_stage"] = (
        backup["symptoms"]["next_stage"].detach().clone()
    )
    data["agent"].symptoms["time_to_next_stage"] = (
        backup["symptoms"]["time_to_next_stage"].detach().clone()
    )
    return data


def make_timer():
    return Timer(
        initial_day="2022-02-01",
        total_days=30,
        weekday_step_duration=(8, 8, 8),
        weekend_step_duration=(
            12,
            12,
        ),
        weekday_activities=(
            ("company", "school", "university", "care_home", "household"),
            ("care_home", "leisure", "household"),
            (
                "care_home",
                "household",
            ),
        ),
        weekend_activities=(
            ("care_home", "leisure", "household"),
            (
                "care_home",
                "household",
            ),
        ),
    )


def group_by_symptoms(symptoms, stages, device):
    current_stage = symptoms["current_stage"]
    ret = torch.zeros(len(stages), device=device)
    for i in range(len(stages)):
        this_stage = current_stage[current_stage == i]
        ret[i] = len(this_stage)
    return ret
