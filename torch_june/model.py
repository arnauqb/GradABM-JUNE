import torch
import yaml
from torch.utils.checkpoint import checkpoint

from torch_june import (
    InfectionPassing,
    TransmissionUpdater,
    IsInfectedSampler,
    SymptomsUpdater,
)
from torch_june.policies import Policies
from torch_june.cuda_utils import get_fraction_gpu_used
from torch_june.paths import default_config_path


class TorchJune(torch.nn.Module):
    def __init__(
        self,
        symptoms_updater=None,
        policies=None,
        infection_passing=None,
        device="cpu",
    ):
        super().__init__()
        if symptoms_updater is None:
            symptoms_updater = SymptomsUpdater.from_file()
        self.symptoms_updater = symptoms_updater
        if policies is None:
            policies = Policies.from_file()
        self.policies = policies
        if infection_passing is None:
            infection_passing = InfectionPassing.from_file()
        self.infection_passing = infection_passing
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
        infection_passing = InfectionPassing.from_parameters(params)
        return cls(
            symptoms_updater=symptoms_updater,
            policies=policies,
            infection_passing=infection_passing,
            device=params["system"]["device"],
        )

    def forward(self, data, timer):
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)
        # not_infected_probs = checkpoint(
        #    self.infection_passing,
        #    data,
        #    timer,
        #    self.policies.interaction_policies,
        #    self.policies.close_venue_policies,
        #    self.policies.quarantine_policies,
        #    use_reentrant=False,
        # )
        not_infected_probs = self.infection_passing(
            data=data,
            timer=timer,
            interaction_policies=self.policies.interaction_policies,
            close_venue_policies=self.policies.close_venue_policies,
            quarantine_policies=self.policies.quarantine_policies,
        )
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
