import torch
import pyro


def infect_fraction_of_people(data, timer, model, fraction_initial_cases):
    n_agents = data["agent"].id.shape[0]
    probs = fraction_initial_cases * torch.ones(n_agents, device=model.device)
    new_infected = pyro.distributions.RelaxedBernoulliStraightThrough(
        temperature=torch.tensor(0.1),
        probs=probs,
    ).rsample()
    data["agent"].susceptibility = torch.maximum(
        torch.tensor(0.0, device=model.device),
        data["agent"].susceptibility - new_infected,
    )
    data["agent"].is_infected = data["agent"].is_infected + new_infected
    data["agent"].infection_time = data["agent"].infection_time + new_infected * (
        timer.now - data["agent"].infection_time
    )
    model.symptoms_updater(
        data=data, timer=timer, new_infected=new_infected
    )


def infect_people_at_indices(data, indices, device="cpu"):
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
