import numpy as np
import torch
from pathlib import Path
from itertools import chain
from torch import distributions as dist
import datetime
import random
from typing import Union
from copy import deepcopy
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from grad_june.paths import grad_june_path
from grad_june.infection import infect_people_at_indices


def read_path(path_str):
    path = Path(path_str)
    if path.parts[0] == "@grad_june":
        path = Path("/".join(path_str.split("/")[1:]))
        path = grad_june_path / path
    return path


def read_date(date: Union[str, datetime.datetime]) -> datetime.datetime:
    """
    Read date in two possible formats, either string or datetime.date, both
    are translated into datetime.datetime to be used by the simulator

    Parameters
    ----------
    date:
        date to translate into datetime.datetime

    Returns
    -------
        date in datetime format
    """
    if type(date) is str:
        return datetime.datetime.strptime(date, "%Y-%m-%d")
    elif isinstance(date, datetime.date):
        return datetime.datetime.combine(date, datetime.datetime.min.time())
    else:
        raise TypeError("date must be a string or a datetime.date object")


def parse_age_probabilities(age_dict, fill_value=0):
    """
    Parses the age probability dictionaries into an array.
    """
    bins = []
    probabilities = []
    for age_range in age_dict:
        age_range_split = age_range.split("-")
        bins.append(int(age_range_split[0]))
        bins.append(int(age_range_split[1]))
        probabilities.append(age_dict[age_range])
    sorting_idx = np.argsort(bins[::2])
    bins = list(
        chain.from_iterable([bins[2 * idx], bins[2 * idx + 1]] for idx in sorting_idx)
    )
    probabilities = np.array(probabilities)[sorting_idx]
    probabilities_binned = []
    for prob in probabilities:
        probabilities_binned.append(fill_value)
        probabilities_binned.append(prob)
    probabilities_binned.append(fill_value)
    probabilities_per_age = []
    for age in range(100):
        idx = np.searchsorted(bins, age + 1)  # we do +1 to include the lower boundary
        probabilities_per_age.append(probabilities_binned[idx])
    return probabilities_per_age


def parse_distribution(dict, device):
    dd = deepcopy(dict)
    dist_name = dd.pop("dist")
    dist_class = getattr(dist, dist_name)
    input = {
        key: torch.tensor(value, device=device, dtype=torch.float)
        for key, value in dd.items()
    }
    return dist_class(**input)


def fix_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Fixing seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_simple_connected_graph(n_agents):
    # avoid circular import
    from grad_june.transmission import TransmissionSampler

    data = HeteroData()
    sampler = TransmissionSampler.from_file()
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
    data["household"].id = torch.zeros(1)
    data["school"].id = torch.zeros(1)
    data["household"].people = torch.tensor([n_agents])
    data["school"].people = torch.tensor([n_agents])
    data["agent", "attends_household", "household"].edge_index = torch.vstack(
        (data["agent"].id[::2], torch.zeros(n_agents//2, dtype=torch.long))
    )
    data["agent", "attends_school", "school"].edge_index = torch.vstack(
        (data["agent"].id[1::2], torch.zeros(n_agents//2, dtype=torch.long))
    )
    data = T.ToUndirected()(data)
    return data
