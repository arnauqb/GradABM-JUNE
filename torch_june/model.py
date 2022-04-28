import torch
from torch.utils.checkpoint import checkpoint

from torch_june import (
    InfectionPassing,
    TransmissionUpdater,
    IsInfectedSampler,
    SymptomsUpdater,
)
from torch_june.policies import Policies
from torch_june.symptoms import SymptomsSampler
from torch_june.cuda_utils import get_fraction_gpu_used


class TorchJune(torch.nn.Module):
    def __init__(
        self,
        symptoms_sampler=None,
        policies=None,
        log_beta_company=torch.tensor(0.0),
        log_beta_school=torch.tensor(0.0),
        log_beta_household=torch.tensor(0.0),
        log_beta_university=torch.tensor(0.0),
        log_beta_leisure=torch.tensor(0.0),
        log_beta_care_home=torch.tensor(0.0),
        device="cpu",
    ):
        super().__init__()
        if symptoms_sampler is None:
            symptoms_sampler = SymptomsSampler.from_default_parameters(device=device)
        if policies is None:
            policies = Policies()
        self.infection_passing = InfectionPassing(
            log_beta_company=log_beta_company.to(device),
            log_beta_school=log_beta_school.to(device),
            log_beta_household=log_beta_household.to(device),
            log_beta_university=log_beta_university.to(device),
            log_beta_leisure=log_beta_leisure.to(device),
            log_beta_care_home=log_beta_care_home.to(device),
        )
        self.transmission_updater = TransmissionUpdater()
        self.is_infected_sampler = IsInfectedSampler()
        self.symptoms_updater = SymptomsUpdater(symptoms_sampler=symptoms_sampler)
        self.policies = policies
        self.device = device

    def forward(self, data, timer):
        # print("FORWARD")
        # print("1")
        # print(get_fraction_gpu_used(6))
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)
        # print("2")
        # print(get_fraction_gpu_used(6))
        # not_infected_probs = checkpoint(self.infection_passing, data, timer)
        not_infected_probs = self.infection_passing(
            data=data,
            timer=timer,
            interaction_policies=self.policies.interaction_policies,
        )
        # print("3")
        # print(get_fraction_gpu_used(6))
        new_infected = self.is_infected_sampler(not_infected_probs)
        # print("4")
        # print(get_fraction_gpu_used(6))
        data["agent"].susceptibility = torch.maximum(
            torch.tensor(0.0, device=self.device),
            data["agent"].susceptibility - new_infected,
        )
        # print("5")
        # print(get_fraction_gpu_used(6))
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        # print("6")
        # print(get_fraction_gpu_used(6))
        data["agent"].infection_time[new_infected.bool()] = timer.now
        # print("7")
        # print(get_fraction_gpu_used(6))
        data["symptoms"] = self.symptoms_updater(
            data=data, timer=timer, new_infected=new_infected
        )
        # print("8")
        # print(get_fraction_gpu_used(6))
        return data
