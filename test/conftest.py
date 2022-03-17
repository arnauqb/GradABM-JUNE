from pathlib import Path
import torch
import torch_geometric.transforms as T
import numpy as np
from torch.distributions import Normal, LogNormal
from pytest import fixture
from torch_geometric.data import HeteroData

from torch_june.infections import InfectionSampler
from torch_june.timer import Timer


@fixture(scope="session", name="june_world_path")
def get_june_world_path():
    return Path(__file__).parent / "data/june_world.hdf5"


@fixture(scope="session", name="june_world_path_only_people")
def get_june_world_path_only_people():
    return Path(__file__).parent / "data/june_world_only_people.hdf5"


@fixture(scope="session", name="sampler")
def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)  # * 1.7
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return InfectionSampler(max_infectiousness, shape, rate, shift)

@fixture(name="agent_data")
def make_agent_data(sampler):
    n_agents = 100
    data = HeteroData()
    data["agent"].id = torch.arange(0, n_agents)
    data["agent"].age = torch.randint(0, 100, (n_agents,))
    data["agent"].sex = torch.randint(0, 2, (n_agents,))
    inf_params = {}
    inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = inf_params_values[0]
    inf_params["shape"] = inf_params_values[1]
    inf_params["rate"] = inf_params_values[2]
    inf_params["shift"] = inf_params_values[3]
    data["agent"].infection_parameters = inf_params
    data["agent"].transmission = torch.zeros(n_agents)
    data["agent"].susceptibility = torch.ones(n_agents)
    data["agent"].is_infected = torch.zeros(n_agents)
    data["agent"].infection_time = -1.0 * torch.ones(n_agents)
    return data

@fixture(name="data")
def make_data(agent_data):
    data = agent_data
    data["school"].id = torch.arange(0, 4)
    data["school"].people = 25 * torch.ones(4)
    data["company"].id = torch.arange(0, 4)
    data["company"].people = 25 * torch.ones(4)
    data["leisure"].id = torch.arange(0, 4)
    data["leisure"].people = 25 * torch.ones(4)
    data["household"].id = torch.arange(0, 25)
    data["household"].people = 4 * torch.ones(25)
    data["agent", "attends_school", "school"].edge_index = torch.vstack(
        (data["agent"].id, torch.tensor(np.repeat(np.arange(0, 4), 25)))
    )
    data["agent", "attends_company", "company"].edge_index = torch.vstack(
        (data["agent"].id, torch.tensor(np.repeat(np.arange(0, 4), 25)))
    )
    data["agent", "attends_leisure", "leisure"].edge_index = torch.vstack(
        (data["agent"].id, torch.tensor(np.repeat(np.arange(0, 4), 25)))
    )
    data["agent", "attends_household", "household"].edge_index = torch.vstack(
        (data["agent"].id, torch.tensor(np.repeat(np.arange(0, 25), 4)))
    )
    data = T.ToUndirected()(data)
    return data


@fixture(name="inf_data")
def make_inf_data(data):
    susc = data["agent"]["susceptibility"].numpy()
    susc[0:100:10] = 0.0
    is_inf = data["agent"]["is_infected"].numpy()
    is_inf[0:100:10] = 1.0
    inf_t = data["agent"]["infection_time"].numpy()
    inf_t[0:100:10] = 0.0
    data["agent"].susceptibility = torch.tensor(susc)
    data["agent"].is_infected = torch.tensor(is_inf)
    data["agent"].infection_time = torch.tensor(inf_t)
    return data


@fixture(name="timer")
def make_timer():
    timer = Timer(
        initial_day="2022-02-01",
        total_days=10,
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
        weekend_activities=(("leisure",), ("household",)),
    )
    return timer

@fixture(name="school_timer")
def make_school_timer():
    timer = Timer(
        initial_day="2022-02-01",
        total_days=10,
        weekday_step_duration=(24,),
        weekend_step_duration=(24,),
        weekday_activities=(("school",),),
        weekend_activities=(("school",),),
    )
    return timer
