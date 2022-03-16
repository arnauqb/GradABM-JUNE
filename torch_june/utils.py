import torch
from torch_geometric.utils import to_undirected
import subprocess


def generate_erdos_renyi(nodes, edge_prob):
    idx = torch.combinations(nodes, r=2)
    mask = torch.rand(idx.size(0)) < edge_prob
    idx = idx[mask]
    edge_index = to_undirected(idx.t(), num_nodes=len(nodes))
    return edge_index


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_fraction_gpu_used(i):
    total = torch.cuda.get_device_properties(f"cuda:{i}").total_memory / 1e6  # MB
    used = get_gpu_memory_map()[i]
    return used / total
