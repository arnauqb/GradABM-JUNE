import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter


class InfectionPassing(MessagePassing):
    def __init__(self, device="cpu"):
        super().__init__(aggr="add", node_dim=-1)
        self.device = device

    def forward(
        self, data, edge_types, betas, transmissions, susceptibilities, delta_time
    ):
        n_agents = len(data["agent"]["id"])
        trans_susc = torch.zeros(n_agents, device=self.device)
        for edge_type in edge_types:
            group_name = "_".join(edge_type.split("_")[1:])
            edge_index = data[edge_type].edge_index
            beta = betas[group_name] * torch.ones(
                len(data[group_name]["id"]), device=self.device
            )
            people_per_group = data[group_name]["people"]
            p_contact = torch.minimum(
                5.0 / people_per_group, torch.tensor(1.0)
            )  # assumes constant n of contacts, change this in the future
            beta = beta * p_contact
            cumulative_trans = self.propagate(edge_index, x=transmissions, y=beta)
            rev_edge_index = data["rev_" + edge_type].edge_index
            trans_susc = trans_susc + self.propagate(
                rev_edge_index, x=cumulative_trans, y=susceptibilities
            )
        return torch.exp(-trans_susc * delta_time)

    def message(self, x_j, y_i):
        return x_j * y_i

    def sample_infected(self, not_infected_probs):
        infected_probs = 1.0 - not_infected_probs
        probs = torch.vstack((infected_probs, not_infected_probs))
        logits = torch.log(probs + 1e-15)
        is_infected = gumbel_softmax(logits, tau=0.1, hard=True, dim=-2)
        return is_infected[0, :]
