import torch

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
        # calculate prob infected by any variant.
        not_infected_total = torch.prod(not_infected_probs, 0)
        logits = torch.log(torch.vstack((not_infected_total, 1.0 - not_infected_probs)))
        infection = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=0.1, hard=True
        )
        is_infected = 1.0 - infection[0, :]
        infection_type = torch.argmax(infection[1:, :], dim=0)
        return is_infected, infection_type


def infect_people(data, timer, new_infected, new_infection_type):
    """
    This function updates the infection status of the agents in the data dictionary.
    It takes as input the data dictionary, the timer, a mask of the newly infected agents
    and the type of infection that they got.

    **Arguments**

    - `data` : PyTorch Geometric data object containing the agent information.
    - `timer` : Timer object containing the current time.
    - `new_infected` : Mask of the newly infected agents.
    - `new_infection_type` : Type of infection that the newly infected agents got.
    """
    data["agent"].susceptibility = torch.clamp(
        data["agent"].susceptibility - new_infected, min=0.0
    )
    data["agent"].is_infected = data["agent"].is_infected + new_infected
    data["agent"].infection_time = data["agent"].infection_time + new_infected * (
        timer.now - data["agent"].infection_time
    )
    data["agent"].infection_id = new_infection_type



def infect_fraction_of_people(
    data, timer, symptoms_updater, fraction, infection_type, device
):
    """
    This function infects a fraction of the population with a given infection type.

    **Arguments**

    - `data` : PyTorch Geometric data object containing the agent information.
    - `timer` : Timer object containing the current time.
    - `symptoms_updater` : SymptomsUpdater object containing the symptoms information.
    - `fraction` : Fraction of the population to infect.
    - `infection_type` : Type of infection to infect the population with.
    - `device` : Device to use for the computation.
    """
    n_infections = data["agent"].susceptibility.shape[0]
    n_agents = data["agent"].id.shape[0]
    probs = torch.zeros((n_infections, n_agents), device=device)
    probs[infection_type, :] = fraction * torch.ones(n_agents, device=device)
    sampler = IsInfectedSampler()
    new_infected, new_infection_type = sampler(
        1.0 - probs
    )  # sampler takes not inf probs
    infect_people(data, timer, new_infected, new_infection_type)
    return new_infected


def infect_people_at_indices(data, indices, device="cpu"):
    """
    Infect people at given indices.

    **Arguments**

    - `data` : PyTorch Geometric data object containing the agent information.
    - `indices` : Indices of the agents to infect.
    - `device` : Device to use for the computation.
    """
    susc = data["agent"]["susceptibility"].cpu().numpy()
    is_inf = data["agent"]["is_infected"].cpu().numpy()
    inf_t = data["agent"]["infection_time"].cpu().numpy()
    next_stage = data["agent"]["symptoms"]["next_stage"].cpu().numpy()
    current_stage = data["agent"]["symptoms"]["current_stage"].cpu().numpy()
    susc[:, indices] = 0.0
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
