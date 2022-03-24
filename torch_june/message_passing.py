import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter


class InfectionPassing(MessagePassing):
    def __init__(self, beta_company, beta_school, beta_household, beta_leisure):
        super().__init__(aggr="add", node_dim=-1)
        #self._betas_to_idcs = {name: i for i, name in enumerate(beta_priors.keys())}
        #for beta_n, beta_v in beta_priors.items():
        #    setattr(self, beta_n, Parameter(beta_v))
        self.beta_company = beta_company
        self.beta_school = beta_school
        self.beta_household = beta_household
        self.beta_leisure = beta_leisure

    def _get_edge_types_from_timer(self, timer):
        ret = []
        for activity in timer.activities:
            ret.append("attends_" + activity)
        return ret

    def forward(self, data, timer):
        edge_types = self._get_edge_types_from_timer(timer)
        delta_time = timer.duration
        n_agents = len(data["agent"]["id"])
        device = data["agent"].transmission.device
        trans_susc = torch.zeros(n_agents, device=device)
        for edge_type in edge_types:
            group_name = "_".join(edge_type.split("_")[1:])
            edge_index = data[edge_type].edge_index
            log_beta = getattr(self, "beta_" + group_name)
            log_beta = log_beta * torch.ones(
                len(data[group_name]["id"]), device=device
            )
            beta = torch.pow(10.0, log_beta)
            people_per_group = data[group_name]["people"]
            p_contact = torch.minimum(
                5.0 / people_per_group, torch.tensor(1.0)
            )  # assumes constant n of contacts, change this in the future
            beta = beta * p_contact
            cumulative_trans = self.propagate(
                edge_index, x=data["agent"].transmission, y=beta
            )
            rev_edge_index = data["rev_" + edge_type].edge_index
            trans_susc = trans_susc + self.propagate(
                rev_edge_index, x=cumulative_trans, y=data["agent"].susceptibility
            )
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
