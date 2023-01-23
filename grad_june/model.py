import torch
import yaml
import pyro
from torch.utils.checkpoint import checkpoint

from grad_june import (
    TransmissionUpdater,
    IsInfectedSampler,
    SymptomsUpdater,
    InfectionNetworks,
)
from grad_june.policies import Policies
from grad_june.cuda_utils import get_fraction_gpu_used
from grad_june.paths import default_config_path


class TorchJune(torch.nn.Module):
    def __init__(
        self,
        symptoms_updater=None,
        policies=None,
        infection_networks=None,
        device="cpu",
    ):
        super().__init__()
        if symptoms_updater is None:
            symptoms_updater = SymptomsUpdater.from_file()
        self.symptoms_updater = symptoms_updater
        if policies is None:
            policies = Policies.from_file()
        self.policies = policies
        if infection_networks is None:
            infection_networks = InfectionNetworks.from_file()
        self.infection_networks = infection_networks
        self.transmission_updater = TransmissionUpdater()
        self.is_infected_sampler = IsInfectedSampler()
        self.device = device

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        symptoms_updater = SymptomsUpdater.from_parameters(params)
        policies = Policies.from_parameters(params)
        infection_networks = InfectionNetworks.from_parameters(params)
        return cls(
            symptoms_updater=symptoms_updater,
            policies=policies,
            infection_networks=infection_networks,
            device=params["system"]["device"],
        )

    def infect_people(self, data, timer, new_infected, new_infection_type):
        data["agent"].susceptibility = torch.clamp(
            data["agent"].susceptibility - new_infected, min=0.0
        )
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        data["agent"].infection_time = data["agent"].infection_time + new_infected * (
            timer.now - data["agent"].infection_time
        )
        data["agent"].infection_id = new_infection_type

    def forward(self, data, timer):
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)
        not_infected_probs = self.infection_networks(
            data=data,
            timer=timer,
            policies=self.policies,
        )
        infected_probs = 1.0 - not_infected_probs
        new_infected, new_infection_type = self.is_infected_sampler(
            not_infected_probs, timer.now
        )
        self.infect_people(data, timer, new_infected, new_infection_type)
        self.symptoms_updater(data=data, timer=timer, new_infected=new_infected)
        return data
