import torch


def infect_fraction_of_people(data, timer, symptoms_updater, fraction, device):
    probs = fraction * torch.ones(data["agent"].id.shape, device=device)
    not_infected_probs = 1.0 - probs.reshape(1,-1)
    not_infected_total = torch.prod(not_infected_probs, 0)
    logits = torch.log(
        torch.vstack((not_infected_total, 1.0 - not_infected_probs))
    )
    infection = torch.nn.functional.gumbel_softmax(
        logits, dim=0, tau=0.1, hard=True
    )
    new_infected = 1.0 - infection[0, :]
    data["agent"].susceptibility = torch.maximum(
        torch.tensor(0.0, device=device),
        data["agent"].susceptibility - new_infected,
    )
    data["agent"].is_infected = data["agent"].is_infected + new_infected
    data["agent"].infection_time = data["agent"].infection_time + new_infected * (
        timer.now - data["agent"].infection_time
    )
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
