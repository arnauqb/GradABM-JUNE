from pathlib import Path
import torch
import torch_geometric.transforms as T
import numpy as np
import random
from pyro.distributions import Normal, LogNormal
from pytest import fixture
from torch_geometric.data import HeteroData

from torch_june.transmission import TransmissionSampler
from torch_june.timer import Timer


@fixture(autouse=True)
def set_random_seed(seed=999):
    """
    Sets global seeds for testing in numpy, random, and numbaized numpy.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return


@fixture(scope="session", name="june_world_path")
def get_june_world_path():
    return Path(__file__).parent / "data/june_world.h5"


@fixture(scope="session", name="sampler")
def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)  # * 1.7
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return TransmissionSampler(max_infectiousness, shape, rate, shift)


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
    data["agent"].infection_time = torch.zeros(n_agents)
    symptoms = {}
    symptoms["current_stage"] = torch.ones(n_agents, dtype=torch.long)
    symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long)
    symptoms["time_to_next_stage"] = torch.zeros(n_agents)
    data["agent"].symptoms = symptoms
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


@fixture(name="person_infector")
def make_person_infector():
    def infector(data, indices):
        susc = data["agent"]["susceptibility"].numpy()
        is_inf = data["agent"]["is_infected"].numpy()
        inf_t = data["agent"]["infection_time"].numpy()
        next_stage = data["agent"]["symptoms"]["next_stage"].numpy()
        susc[indices] = 0.0
        is_inf[indices] = 1.0
        inf_t[indices] = 0.0
        next_stage[indices] = 2
        data["agent"]["susceptibility"] = torch.tensor(susc)
        data["agent"]["is_infected"] = torch.tensor(is_inf)
        data["agent"]["infection_time"] = torch.tensor(inf_t)
        data["agent"]["symptoms"]["next_stage"] = torch.tensor(next_stage)
        return data

    return infector


@fixture(name="inf_data")
def make_inf_data(data, person_infector):
    return person_infector(data, list(range(0, 100, 10)))


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
            ("company", "school", "household"),
            ("leisure", "household"),
            ("household",),
        ),
        weekend_activities=(
            ("leisure",),
            ("household",),
        ),
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
