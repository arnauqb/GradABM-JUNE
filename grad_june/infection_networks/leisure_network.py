import torch

from grad_june.infection_networks import InfectionNetwork
from grad_june.utils import parse_age_probabilities


class LeisureNetwork(InfectionNetwork):
    def __init__(self, log_beta, leisure_probabilities, device):
        super().__init__(log_beta=log_beta, device=device)
        self.leisure_probabilities = self._parse_leisure_probabilities(
            leisure_probabilities
        )
        self.weekday_probabilities = None
        self.weekend_probabilities = None

    @classmethod
    def from_parameters(cls, params):
        device = params["system"]["device"]
        leisure_probabilities = params["leisure"][cls._get_name()]
        return cls(
            device=device,
            leisure_probabilities=leisure_probabilities,
            **params["networks"][cls._get_name()]
        )

    def _parse_leisure_probabilities(self, leisure_probabilities):
        ret = torch.zeros((2, 2, 100), device=self.device)
        for i, day_type in enumerate(["weekday", "weekend"]):
            for j, sex in enumerate(["male", "female"]):
                parsed_probs = parse_age_probabilities(
                    leisure_probabilities[day_type][sex]
                )
                ret[i, j, :] = torch.tensor(parsed_probs, device=self.device)
        return ret

    def initialize_leisure_probabilities(self, data):
        self.weekday_probabilities = self.leisure_probabilities[
            0, data["agent"].sex, data["agent"].age
        ]
        self.weekend_probabilities = self.leisure_probabilities[
            1, data["agent"].sex, data["agent"].age
        ]

    def _get_edge_index(self, data):
        return data["attends_leisure"].edge_index

    def _get_reverse_edge_index(self, data):
        return data["rev_attends_leisure"].edge_index

    def _get_beta(self, policies, timer, data):
        interaction_policies = policies.interaction_policies
        beta = 10.0**self.log_beta
        if interaction_policies:
            beta = interaction_policies.apply(beta=beta, name=self.name, timer=timer)
        beta = beta * torch.ones(len(data["leisure"]["id"]), device=self.device)
        return beta

    def _get_people_per_group(self, data):
        return data["leisure"]["people"]

    def _get_transmissions(self, data, policies, timer):
        if self.weekday_probabilities is None:
            self.initialize_leisure_probabilities(data)
        if policies.quarantine_policies:
            mask = policies.quarantine_policies.quarantine_mask
        else:
            mask = 1.0
        if timer.day_type == "weekday":
            leisure_mask = self.weekday_probabilities
        else:
            leisure_mask = self.weekend_probabilities
        return mask * leisure_mask * data["agent"].transmission

    def _get_susceptibilities(self, data, policies, timer):
        if self.weekday_probabilities is None:
            self.initialize_leisure_probabilities(data)
        if policies.quarantine_policies:
            mask = policies.quarantine_policies.quarantine_mask
        else:
            mask = 1.0
        if timer.day_type == "weekday":
            leisure_mask = self.weekday_probabilities
        else:
            leisure_mask = self.weekend_probabilities
        return mask * leisure_mask * data["agent"].susceptibility


class PubNetwork(LeisureNetwork):
    pass


class CinemaNetwork(LeisureNetwork):
    pass


class GroceryNetwork(LeisureNetwork):
    pass


class GymNetwork(LeisureNetwork):
    pass


class VisitNetwork(LeisureNetwork):
    pass

class CareVisitNetwork(LeisureNetwork):
    def _get_susceptibilities(self, data, policies, timer):
        mask_age = data["agent"].age > 75
        if self.weekday_probabilities is None:
            self.initialize_leisure_probabilities(data)
        if policies.quarantine_policies:
            mask = policies.quarantine_policies.quarantine_mask
        else:
            mask = 1.0
        if timer.day_type == "weekday":
            leisure_mask = self.weekday_probabilities
        else:
            leisure_mask = self.weekend_probabilities
        return mask * leisure_mask * data["agent"].susceptibility * mask_age
