import torch
from torch.nn.parameter import Parameter
from time import time

from torch_june import InfectionPassing, InfectionUpdater, IsInfectedSampler
from torch_june.cuda_utils import get_fraction_gpu_used


class TorchJune(torch.nn.Module):
    def __init__(self, beta_company, beta_leisure, beta_school, beta_household):
        super().__init__()
        self.infection_passing = InfectionPassing(
            beta_company=beta_company,
            beta_household=beta_household,
            beta_leisure=beta_leisure,
            beta_school=beta_school,
        )
        self.infection_updater = InfectionUpdater()
        self.is_infected_sampler = IsInfectedSampler()

    def forward(self, data, timer):
        data["agent"].transmission = self.infection_updater(data=data, timer=timer)
        not_infected_probs = self.infection_passing(data=data, timer=timer)
        new_infected = self.is_infected_sampler(not_infected_probs)
        data["agent"].susceptibility = data["agent"].susceptibility - new_infected
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        data["agent"].infection_time = data["agent"].infection_time + new_infected * (
            1.0 + timer.now
        )
        return data
