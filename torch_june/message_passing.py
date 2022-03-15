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
        trans_susc = torch.zeros(
            len(data["agent"].id), device=self.device#, requires_grad=True
        )
        for edge_type in edge_types:
            group_name = "_".join(edge_type.split("_")[1:])
            edge_index = data[edge_type].edge_index
            if edge_type == "attends_leisure":
                beta = betas[group_name] * torch.ones(
                    len(data["agent"]["id"]), device=self.device#, requires_grad=True
                )
            else:
                beta = betas[group_name] * torch.ones(
                    len(data[group_name]["id"]), device=self.device#, requires_grad=True
                )
            cumulative_trans = self.propagate(edge_index, x=transmissions, y=beta)
            rev_edge_index = data["rev_" + edge_type].edge_index
            trans_susc = trans_susc + self.propagate(
                rev_edge_index, x=cumulative_trans, y=susceptibilities
            )
        return torch.exp(-trans_susc * delta_time)

    def message(self, x_j, y_i):
        return x_j * y_i

    def sample_infected(self, infected_probs):
        no_infected_probs = 1.0 - infected_probs
        probs = torch.vstack((infected_probs, no_infected_probs))
        logits = torch.log(probs + 1e-15)
        is_infected = gumbel_softmax(logits, tau=0.1, hard=True, dim=-2)
        return is_infected[1, :]
