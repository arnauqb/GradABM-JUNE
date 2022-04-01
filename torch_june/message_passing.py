import torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import gumbel_softmax
from pyro.distributions import RelaxedBernoulliStraightThrough

activity_hierarchy = [
    "attends_school",
    "attends_university",
    "attends_company",
    "attends_care_home",
    "attends_leisure",
    "attends_household",
]

import pickle


class InfectionPassing(MessagePassing):
    def __init__(
        self,
        beta_company=torch.tensor(1.0),
        beta_school=torch.tensor(1.0),
        beta_household=torch.tensor(1.0),
        beta_university=torch.tensor(1.0),
        beta_leisure=torch.tensor(1.0),
        beta_care_home=torch.tensor(1.0),
    ):
        super().__init__(aggr="add", node_dim=-1)
        self.beta_company = beta_company
        self.beta_school = beta_school
        self.beta_household = beta_household
        self.beta_leisure = beta_leisure
        self.beta_university = beta_university
        self.beta_care_home = beta_care_home

    def _get_edge_types_from_timer(self, timer):
        ret = []
        for activity in timer.activities:
            ret.append("attends_" + activity)
        return ret

    def _apply_activity_hierarchy(self, activities):
        """
        Returns a list of activities with the right order,
        obeying the permanent activity hierarcy and shuflling
        the random one.

        Parameters
        ----------
        activities:
            list of activities that take place at a given time step
        Returns
        -------
        Ordered list of activities according to hierarchy
        """
        activities.sort(key=lambda x: activity_hierarchy.index(x))
        return activities

    def forward(self, data, timer):
        edge_types = self._get_edge_types_from_timer(timer)
        edge_types = self._apply_activity_hierarchy(edge_types)
        delta_time = timer.duration
        n_agents = len(data["agent"]["id"])
        device = data["agent"].transmission.device
        trans_susc = torch.zeros(n_agents, device=device)

        is_free = torch.ones(n_agents, device=device)
        for edge_type in edge_types:
            group_name = "_".join(edge_type.split("_")[1:])
            edge_index = data[edge_type].edge_index
            beta = getattr(self, "beta_" + group_name)
            beta = beta * torch.ones(len(data[group_name]["id"]), device=device)
            people_per_group = data[group_name]["people"]
            p_contact = torch.maximum(
                torch.minimum(
                    1.0 / (people_per_group - 1), torch.tensor(1.0, device=device)
                ),
                torch.tensor(0.0, device=device),
            )  # assumes constant n of contacts, change this in the future
            beta = beta * p_contact
            # remove people who are not really in this group
            transmissions = data["agent"].transmission * is_free
            cumulative_trans = self.propagate(edge_index, x=transmissions, y=beta)
            rev_edge_index = data["rev_" + edge_type].edge_index
            # people who are not here can't be infected.
            susceptibilities = data["agent"].susceptibility * is_free
            trans_susc = trans_susc + self.propagate(
                rev_edge_index, x=cumulative_trans, y=susceptibilities
            )
            mask = torch.ones(n_agents, dtype=torch.int, device=device)
            mask[edge_index[0, :]] = 0
            is_free = is_free * mask
        trans_susc = torch.clamp(
            trans_susc, min=1e-6
        )  # this is necessary to avoid gradient infs
        not_infected_probs = torch.exp(-trans_susc * delta_time)
        return not_infected_probs

    def message(self, x_j, y_i):
        return x_j * y_i


class IsInfectedSampler(torch.nn.Module):
    def forward(self, not_infected_probs):
        infected_probs = 1.0 - not_infected_probs
        dist = RelaxedBernoulliStraightThrough(temperature=0.1, probs=infected_probs)
        return dist.rsample()

        # probs = torch.vstack((infected_probs, not_infected_probs))
        # logits = torch.log(probs + 1e-15)
        # is_infected = gumbel_softmax(logits, tau=0.1, hard=True, dim=-2)
        # return is_infected[0, :]
