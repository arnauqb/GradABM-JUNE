import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter

activity_hierarchy = [
    "attends_school",
    "attends_company",
    "attends_leisure",
    "attends_household",
]


class InfectionPassing(MessagePassing):
    def __init__(
        self,
        log_beta_company=torch.tensor(-10.0),
        log_beta_school=torch.tensor(-10.0),
        log_beta_household=torch.tensor(-10.0),
        log_beta_leisure=torch.tensor(-10.0),
    ):
        super().__init__(aggr="add", node_dim=-1)
        self.log_beta_company = log_beta_company
        self.log_beta_school = log_beta_school
        self.log_beta_household = log_beta_household
        self.log_beta_leisure = log_beta_leisure

    def _get_edge_types_from_timer(self, timer):
        ret = []
        for activity in timer.activities:
            ret.append("attends_" + activity)
        return ret

    def _apply_activity_hierarchy(self, activities):
        """
        Returns a list of activities with the right order, obeying the permanent activity hierarcy
        and shuflling the random one.

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
            log_beta = getattr(self, "log_beta_" + group_name)
            log_beta = log_beta * torch.ones(len(data[group_name]["id"]), device=device)
            beta = torch.pow(10.0, log_beta)
            people_per_group = data[group_name]["people"]
            p_contact = torch.minimum(
                1.0 / (people_per_group - 1), torch.tensor(1.0)
            )  # assumes constant n of contacts, change this in the future
            beta = beta * p_contact
            # remove people who are not really in this group
            transmissions = data["agent"].transmission * is_free
            cumulative_trans = self.propagate(
                edge_index, x=transmissions, y=beta
            )
            rev_edge_index = data["rev_" + edge_type].edge_index
            # people who are not here can't be infected.
            susceptibilities = data["agent"].susceptibility * is_free
            trans_susc = trans_susc + self.propagate(
                rev_edge_index, x=cumulative_trans, y=susceptibilities
            )
            is_free[edge_index[0,:]] = 0.0
        not_infected_probs = torch.exp(-trans_susc * delta_time)
        return not_infected_probs

    def message(self, x_j, y_i):
        return x_j * y_i


class IsInfectedSampler(torch.nn.Module):
    def forward(self, not_infected_probs):
        infected_probs = 1.0 - not_infected_probs
        probs = torch.vstack((infected_probs, not_infected_probs))
        logits = torch.log(probs + 1e-15)
        is_infected = gumbel_softmax(logits, tau=0.1, hard=True, dim=-2)
        return is_infected[0, :]
