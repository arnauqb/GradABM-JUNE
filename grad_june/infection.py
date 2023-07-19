import torch
import torch_geometric

from grad_june.demographics import get_people_per_area

class IsInfectedSampler(torch.nn.Module):
    def forward(self, not_infected_probs):
        """
        Here we need to sample the infection status of each agent and the variant that
        the agent gets in case of infection.
        To do this, we construct a tensor of size [M+1, N], where M is the number of
        variants and N the number of agents. The extra dimension in M will represent
        the agent not getting infected, so that it can be sampled as an outcome using
        the Gumbel-Softmax reparametrization of the categorical distribution.
        """
        logits = torch.vstack((not_infected_probs, 1.0 - not_infected_probs)).log()
        infection = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=0.1, hard=True
        )
        is_infected = 1.0 - infection[0, :]
        return is_infected


def infect_people(data, timer, new_infected):
    data["agent"].susceptibility = torch.clamp(
        data["agent"].susceptibility - new_infected, min=0.0
    )
    data["agent"].is_infected = data["agent"].is_infected + new_infected
    data["agent"].infection_time = data["agent"].infection_time + new_infected * (
        timer.now - data["agent"].infection_time
    )


def infect_fraction_of_people(
    data, timer, symptoms_updater, fraction, device
):
    n_agents = data["agent"].id.shape[0]
    probs = fraction * torch.ones(n_agents, device=device)
    sampler = IsInfectedSampler()
    new_infected = sampler(
        1.0 - probs
    )  # sampler takes not inf probs
    infect_people(data, timer, new_infected)
    return new_infected


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

def infect_people_at_districts(data: torch_geometric.data.HeteroData, cases_per_district: dict):
    """
    Infects people at districts. The number of cases per district is given in the `cases_per_district` dictionary.

    **Arguments:**

    - `data`: The data object containing the agent data.
    - `cases_per_district`: A dictionary mapping district ids to the number of cases in that district.
    """
    people_per_district = get_people_per_area(data["agent"].id, data["agent"].district_id)
    ids_to_infect = []
    for district in people_per_district:
        n_to_infect = cases_per_district.get(district, 0)
        random_idcs = torch.randperm(len(people_per_district[district]))[:n_to_infect]
        agent_ids = people_per_district[district][random_idcs]
        ids_to_infect.extend(list(agent_ids.cpu().numpy()))
    infect_people_at_indices(data, ids_to_infect)