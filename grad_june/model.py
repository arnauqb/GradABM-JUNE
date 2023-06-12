import torch
import yaml
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


class GradJune(torch.nn.Module):
    """
    This class represents an epidemiological simulation model.

    Attributes:
        symptoms_updater: An object that updates agents' symptoms based on their infection status.
        policies: An object that contains the current policies used for the simulation.
        infection_networks: An object that calculates the probability of not being infected for each agent based on current policies.
        transmission_updater: An object that updates agent transmission based on current transmission updater values.
        is_infected_sampler: An object that samples which agents will be infected based on their not_infected probabilities.
        device: A string representing the device being used for the simulation.
    """

    def __init__(
        self,
        symptoms_updater=None,
        policies=None,
        infection_networks=None,
        device="cpu",
    ):
        super().__init__()

        # Initializes symptoms updater, policies, and infection networks.
        if symptoms_updater is None:
            symptoms_updater = SymptomsUpdater.from_file()
        self.symptoms_updater = symptoms_updater
        if policies is None:
            policies = Policies.from_file()
        self.policies = policies
        if infection_networks is None:
            infection_networks = InfectionNetworks.from_file()
        self.infection_networks = infection_networks

        # Initializes transmission updater, is_infected_sampler, and device.
        self.transmission_updater = TransmissionUpdater()
        self.is_infected_sampler = IsInfectedSampler()
        self.device = device

    @classmethod
    def from_file(cls, fpath=default_config_path):
        """
        This method creates a new instance of the GradJune class from a configuration file.

        Args:
            fpath: A string representing the path to the configuration file.

        Returns:
            A new instance of the GradJune class.
        """
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        """
        This method creates a new instance of the GradJune class from a dictionary of parameters.

        Args:
            params: A dictionary containing the necessary parameters.

        Returns:
            A new instance of the GradJune class.
        """
        symptoms_updater = SymptomsUpdater.from_parameters(params)
        policies = Policies.from_parameters(params)
        infection_networks = InfectionNetworks.from_parameters(params)
        return cls(
            symptoms_updater=symptoms_updater,
            policies=policies,
            infection_networks=infection_networks,
            device=params["system"]["device"],
        )

    def infect_people(self, data, timer, new_infected):
        """
        This method infects people based on the given parameters.

        Args:
            data: A dictionary containing simulation data.
            timer: An integer representing the current simulation time.
            new_infected: A tensor representing which agents will be infected.

        Returns:
            None.
        """
        # Updates agent susceptibility, infection status, and infection time based on new_infected tensor.
        data["agent"].susceptibility = torch.maximum(
            torch.tensor(0.0, device=self.device),
            data["agent"].susceptibility - new_infected,
        )
        data["agent"].is_infected = data["agent"].is_infected + new_infected
        data["agent"].infection_time = data["agent"].infection_time + new_infected * (
            timer.now - data["agent"].infection_time
        )

    def forward(self, data, timer):
        """
        This function represents a forward pass through the epidemiological simulation model.

        Args:
            data: A PyTorch geometric data object containing simulation data.
            timer: An integer representing the current simulation time.

        Returns:
            A A PyTorch geometric data object containing updated simulation data.
        """

        # Updates agent transmission based on current transmission updater values.
        data["agent"].transmission = self.transmission_updater(data=data, timer=timer)

        # Calculates probability of not being infected for each agent based on current policies.
        not_infected_probs = self.infection_networks(
            data=data,
            timer=timer,
            policies=self.policies,
        )

        # Samples which agents will be infected based on their not_infected probabilities.
        new_infected = self.is_infected_sampler(not_infected_probs)

        # Infects agents who were sampled as new_infected.
        self.infect_people(data, timer, new_infected)

        # Updates agents' symptoms based on their infection status.
        self.symptoms_updater(data=data, timer=timer, new_infected=new_infected)

        # Returns updated simulation data.
        return data
