import torch
from torch_geometric.data import HeteroData

from grad_june.demographics import get_people_per_area
from grad_june.infection import IsInfectedSampler
from grad_june.timer import Timer

def infect_people(data: HeteroData, time: int, new_infected: torch.Tensor):
    data["agent"].susceptibility = torch.clamp(
        data["agent"].susceptibility - new_infected, min=0.0
    )
    data["agent"].is_infected = data["agent"].is_infected + new_infected
    data["agent"].infection_time = data["agent"].infection_time + new_infected * (
        time - data["agent"].infection_time
    )

def infect_people_at_indices(data, indices, device="cpu"):
    susc = data["agent"]["susceptibility"].cpu().numpy()
    is_inf = data["agent"]["is_infected"].cpu().numpy()
    inf_t = data["agent"]["infection_time"].cpu().numpy()
    next_stage = data["agent"]["symptoms"]["next_stage"].cpu().numpy()
    current_stage = data["agent"]["symptoms"]["current_stage"].cpu().numpy()
    susc[indices] = 0.0
    is_inf[indices] = 1.0
    inf_t[indices] = 0.0
    next_stage[indices] = 2
    current_stage[indices] = 1
    data["agent"]["susceptibility"] = torch.tensor(susc, device=device)
    data["agent"]["is_infected"] = torch.tensor(is_inf, device=device)
    data["agent"]["infection_time"] = torch.tensor(inf_t, device=device)
    data["agent"]["symptoms"]["next_stage"] = torch.tensor(next_stage, device=device)
    data["agent"]["symptoms"]["current_stage"] = torch.tensor(
        current_stage, device=device
    )
    return data


class InfectionSeedByDistrict(torch.nn.Module):
    def __init__(self, cases_per_district: dict, device: str):
        """
        Seeds infections at districts. The number of cases per district is given in the `cases_per_district` dictionary.

        **Arguments:**

        - `cases_per_district`: a dictionary mapping district ids to a list of cases per time step
        - `device`: the device to use for the tensor operations
        """
        super().__init__()
        self.cases_per_district = cases_per_district
        self.device = device

    def forward(
        self, data: HeteroData, time_step: int
    ):
        people_per_district = get_people_per_area(
            data["agent"].id, data["agent"].district_id
        )
        ids_to_infect = []
        for district in people_per_district:
            if district in self.cases_per_district:
                n_to_infect = self.cases_per_district[district][time_step]
            else:
                n_to_infect = 0
            people = people_per_district[district]
            infected_status = data["agent"].is_infected[people]
            susceptible_people = people[~infected_status.bool()]
            random_idcs = torch.randperm(len(susceptible_people))[:n_to_infect]
            agent_ids = susceptible_people[random_idcs]
            ids_to_infect.extend(list(agent_ids.cpu().numpy()))
        infect_people_at_indices(data, ids_to_infect)

class InfectionSeedByFraction(torch.nn.Module):
    def __init__(self, log_fraction: float, device: str):
        """
        Infects a fraction of the population at random

        **Arguments:**

        - `log_fraction`: the fraction of the population to infect in log10
        - `device`: the device to use for the tensor operations
        """
        super().__init__()
        self.log_fraction = log_fraction
        self.device = device

    def forward(
        self, data: HeteroData, time_step: int
    ):
        if time_step > 0:
            return
        n_agents = data["agent"].id.shape[0]
        fraction = 10 ** self.log_fraction
        probs = fraction * torch.ones(n_agents, device=self.device)
        sampler = IsInfectedSampler()
        new_infected = sampler(1.0 - probs)  # sampler takes not inf probs
        infect_people(data, time_step, new_infected)

def get_seed_from_parameters(params: dict, device: str):
    seed_type = params["infection_seed"]["type"]
    params = params["infection_seed"]["params"]
    return eval(seed_type)(**params, device=device)