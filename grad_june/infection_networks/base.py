import re
import yaml
import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing

from grad_june.paths import default_config_path
import grad_june.infection_networks


class InfectionNetwork(MessagePassing):
    def __init__(self, log_beta, device="cpu"):
        super().__init__( aggr="add", node_dim=-1)
        self.device = device
        if type(log_beta) != torch.nn.Parameter:
            self.log_beta = torch.tensor(float(log_beta))
        else:
            self.log_beta = log_beta
        self.name = self._get_name()

    @classmethod
    def from_parameters(cls, params):
        device = params["system"]["device"]
        return cls(device=device, **params["networks"][cls._get_name()])

    @classmethod
    def _get_name(cls):
        return "_".join(re.findall("[A-Z][^A-Z]*", cls.__name__)[:-1]).lower()

    def _get_edge_index(self, data):
        return data["attends_" + self.name].edge_index

    def _get_reverse_edge_index(self, data):
        return data["rev_attends_" + self.name].edge_index

    def _get_beta(self, policies, timer, data):
        interaction_policies = policies.interaction_policies
        beta = 10.0**self.log_beta
        if interaction_policies:
            beta = interaction_policies.apply(beta=beta, name=self.name, timer=timer)
        beta = beta * torch.ones(len(data[self.name]["id"]), device=self.device)
        return beta

    def _get_people_per_group(self, data):
        return data[self.name]["people"]

    def _get_transmissions(self, data, policies, timer):
        if policies.quarantine_policies:
            mask = policies.quarantine_policies.quarantine_mask
        else:
            mask = 1.0
        return mask * data["agent"].transmission

    def _get_susceptibilities(self, data, policies, timer):
        if policies.quarantine_policies:
            mask = policies.quarantine_policies.quarantine_mask
        else:
            mask = 1.0
        return mask * data["agent"].susceptibility

    def forward(self, data, timer, policies):
        beta = self._get_beta(policies=policies, timer=timer, data=data)
        people_per_group = self._get_people_per_group(data)
        p_contact = torch.maximum(
            torch.minimum(
                1.0 / (people_per_group - 1), torch.tensor(1.0, device=self.device)
            ),
            torch.tensor(0.0, device=self.device),
        )  # assumes constant n of contacts, change this in the future
        beta = beta * p_contact
        # remove people who are not really in this group
        transmissions = self._get_transmissions(
            data=data, policies=policies, timer=timer
        )
        susceptibilities = self._get_susceptibilities(
            data=data, policies=policies, timer=timer
        )
        edge_index = self._get_edge_index(data)
        cumulative_trans = self.propagate(edge_index, x=transmissions, y=beta)
        rev_edge_index = self._get_reverse_edge_index(data)
        trans_susc = self.propagate(
            rev_edge_index, x=cumulative_trans, y=susceptibilities
        )
        return trans_susc

    def message(self, x_j, y_i):
        return x_j * y_i


class InfectionNetworks(torch.nn.Module):
    def __init__(self, device="cpu", **kwargs):
        super().__init__()
        self.networks = torch.nn.ModuleDict(kwargs)
        self.device = device

    def __getitem__(self, item):
        return self.networks[item]

    @classmethod
    def from_parameters(cls, params):
        device = params["system"]["device"]
        network_params = params["networks"]
        network_dict = {}
        for key in network_params:
            network_name = "".join(word.title() for word in key.split("_"))
            network_name = network_name + "Network"
            network_class = getattr(grad_june.infection_networks, network_name)
            network = network_class.from_parameters(params)
            network_dict[key] = network
        return cls(device=device, **network_dict)

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    def forward(
        self,
        data,
        timer,
        policies,
    ):
        n_agents = len(data["agent"].id)
        delta_time = timer.duration
        policies.apply(timer=timer, data=data)
        trans_susc = torch.zeros(n_agents, device=self.device)
        activity_order = timer.get_activity_order()
        if policies.close_venue_policies:
            activity_order = policies.close_venue_policies.apply(
                edge_types=activity_order, timer=timer
            )
        for activity in activity_order:
            network = self.networks[activity]
            trans_susc += network(data=data, timer=timer, policies=policies)
        trans_susc = torch.clamp(
            trans_susc, min=1e-6, max = 100
        )  # this is necessary to avoid gradient nans
        not_infected_probs = torch.exp(-trans_susc * delta_time)
        not_infected_probs = torch.clamp(not_infected_probs, min=0.0, max=1.0)
        return not_infected_probs


class HouseholdNetwork(InfectionNetwork):
    def _get_transmissions(self, data, policies, timer):
        return data["agent"].transmission

    def _get_susceptibilities(self, data, policies, timer):
        return data["agent"].susceptibility

    pass


class CareHomeNetwork(InfectionNetwork):
    pass


class SchoolNetwork(InfectionNetwork):
    pass


class CompanyNetwork(InfectionNetwork):
    pass


class UniversityNetwork(InfectionNetwork):
    pass

