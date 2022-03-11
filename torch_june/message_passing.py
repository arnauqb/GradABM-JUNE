import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn.functional import gumbel_softmax



class InfectionPassing(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", node_dim=-1)

    def forward(self, data, edge_types):
        n_agents = len(data["agent"]["id"])
        ret = {}
        for edge_type in edge_types:
            edge_index = data[edge_type].edge_index
            eff_beta = data["school"].beta
            transmissions = self.propagate(
                edge_index, x=data["agent"]["transmission"], y=eff_beta
            )
            rev_edge_index = data["rev_" + edge_type].edge_index
            trans_susc = self.propagate(
                rev_edge_index, x=transmissions, y=data["agent"].susceptibility
            )
            probabilities = torch.exp(-trans_susc)
            ret[edge_type] = probabilities
        return ret

    def message(self, x_j, y_i):
        return x_j * y_i

    def sample_infected(self, infected_probs):
        no_infected_probs = 1.0 - infected_probs
        probs = torch.vstack((infected_probs, no_infected_probs))
        logits = torch.log(probs)
        is_infected = gumbel_softmax(logits, tau=1, hard=True, dim=-2)
        return is_infected[0,:]



