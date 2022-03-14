import torch
from torch_geometric.utils import to_undirected


def generate_erdos_renyi(nodes, edge_prob):
    idx = torch.combinations(nodes, r=2)
    mask = torch.rand(idx.size(0)) < edge_prob
    idx = idx[mask]
    edge_index = to_undirected(idx.t(), num_nodes=len(nodes))
    return edge_index

