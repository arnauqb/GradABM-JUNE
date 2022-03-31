import torch
from torch.nn.parameter import Parameter
from time import time

from torch_june import (
    InfectionPassing,
    TransmissionUpdater,
    IsInfectedSampler,
    SymptomsUpdater,
)
from torch_june.cuda_utils import get_fraction_gpu_used
from torch_june.symptoms import SymptomsSampler


class TorchJune(torch.nn.Module):
    def __init__(
        self,
        symptoms_sampler=None,
        beta_company=torch.tensor(1.0),
        beta_school=torch.tensor(1.0),
        beta_household=torch.tensor(1.0),
        beta_university=torch.tensor(1.0),
        beta_leisure=torch.tensor(1.0),
        beta_care_home=torch.tensor(1.0),
    ):
        if symptoms_sampler is None:
            symptoms_sampler = SymptomsSampler.from_default_parameters()
        super().__init__()
        self.infection_passing = InfectionPassing(
            beta_company=beta_company,
            beta_school=beta_school,
            beta_household=beta_household,
            beta_university=beta_university,
            beta_leisure=beta_leisure,
            beta_care_home=beta_care_home,
        )
        self.transmission_updater = TransmissionUpdater()
        self.is_infected_sampler = IsInfectedSampler()
        self.symptoms_updater = SymptomsUpdater(symptoms_sampler=symptoms_sampler)

    def forward(self, data, timer):
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)
        not_infected_probs = self.infection_passing(data=data, timer=timer)
        new_infected = self.is_infected_sampler(not_infected_probs)
        data["agent"].susceptibility = data["agent"].susceptibility - new_infected
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        data["agent"].infection_time[new_infected.bool()] = timer.now
        data["symptoms"] = self.symptoms_updater(
            data=data, timer=timer, new_infected=new_infected
        )
        return data
