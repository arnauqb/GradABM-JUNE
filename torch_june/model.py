import torch

from torch_june import (
    InfectionPassing,
    TransmissionUpdater,
    IsInfectedSampler,
    SymptomsUpdater,
)
from torch_june.symptoms import SymptomsSampler


class TorchJune(torch.nn.Module):
    def __init__(
        self,
        symptoms_sampler=None,
        log_beta_company=torch.tensor(0.0),
        log_beta_school=torch.tensor(0.0),
        log_beta_household=torch.tensor(0.0),
        log_beta_university=torch.tensor(0.0),
        log_beta_leisure=torch.tensor(0.0),
        log_beta_care_home=torch.tensor(0.0),
        device="cpu",
    ):
        if symptoms_sampler is None:
            symptoms_sampler = SymptomsSampler.from_default_parameters(device=device)
        super().__init__()
        self.infection_passing = InfectionPassing(
            log_beta_company=log_beta_company,
            log_beta_school=log_beta_school,
            log_beta_household=log_beta_household,
            log_beta_university=log_beta_university,
            log_beta_leisure=log_beta_leisure,
            log_beta_care_home=log_beta_care_home,
        )
        self.transmission_updater = TransmissionUpdater()
        self.is_infected_sampler = IsInfectedSampler()
        self.symptoms_updater = SymptomsUpdater(symptoms_sampler=symptoms_sampler)
        self.device = device

    def forward(self, data, timer):
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)
        not_infected_probs = self.infection_passing(data=data, timer=timer)
        new_infected = self.is_infected_sampler(not_infected_probs)
        data["agent"].susceptibility = torch.maximum(
            torch.tensor(0.0, device=self.device),
            data["agent"].susceptibility - new_infected,
        )
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        data["agent"].infection_time[new_infected.bool()] = timer.now
        data["symptoms"] = self.symptoms_updater(
            data=data, timer=timer, new_infected=new_infected
        )
        return data
