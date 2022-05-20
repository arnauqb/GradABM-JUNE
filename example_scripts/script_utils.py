import torch
import numpy as np
import pickle
import random
from pyro.distributions import Normal, LogNormal

from torch_june import TransmissionSampler, Timer, TorchJune
from torch_june.policies import Policies

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



def group_by_symptoms(symptoms, stages, device):
    current_stage = symptoms["current_stage"]
    ret = torch.zeros(len(stages), device=device)
    for i in range(len(stages)):
        this_stage = current_stage[current_stage == i]
        ret[i] = len(this_stage)
    return ret


def get_people_by_age(data, device):
    ages = torch.tensor([0, 18, 25, 65, 80], device=device)
    ret = torch.zeros(ages.shape[0] - 1, device=device)
    for i in range(1, ages.shape[0]):
        mask1 = data["agent"].age < ages[i]
        mask2 = data["agent"].age > ages[i - 1]
        mask = mask1 * mask2
        ret[i - 1] = mask.sum()
    return ret

