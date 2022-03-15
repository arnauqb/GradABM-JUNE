from torch_june import GraphLoader, InfectionPassing, AgentDataLoader, TorchJune, Timer
from torch_geometric.data import HeteroData
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import h5py
import networkx
from torch_june.june_world_loader import LeisureNetworkLoader
from torch_june.infections import Infections, InfectionSampler
from torch_june.utils import generate_erdos_renyi
from torch.distributions import Normal, LogNormal
from time import time

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"


betas = {"company": 1.0, "school": 2.0, "household": 3.0, "leisure": 1.0}

#june_world_path = "/cosma7/data/dp004/dc-quer1/JUNE
#june_world_path = "/cosma/home/dp004/dc-quer1/june/JUNE/example_scripts/tests.hdf5"
#data = HeteroData()
#data = GraphLoader(june_world_path).load_graph(data)
#AgentDataLoader(june_world_path).load_agent_data(data)
with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

max_infectiousness = LogNormal(0, 0.5)  # * 1.7
shape = Normal(1.56, 0.08)
rate = Normal(0.53, 0.03)
shift = Normal(-2.12, 0.1)
sampler = InfectionSampler(max_infectiousness, shape, rate, shift)


initial_infected = np.zeros(len(data["agent"]["id"]))
initial_infected[np.random.randint(0, len(initial_infected), size=100)] = 1
initial_infected = torch.tensor(initial_infected)
infections = Infections(
    sampler(len(data["agent"]["id"])), initial_infected=initial_infected, device=device
)
model = TorchJune(data=data, betas=betas, infections=infections, device=device)

susc = np.ones(len(data["agent"]["id"]))
susc[0] = 0.0
susc = torch.tensor(susc, requires_grad=True).to(device)

timer = Timer(
    initial_day="2022-03-18",
    total_days=15,
    weekday_step_duration=(8, 8, 8),
    weekend_step_duration=(
        12,
        12,
    ),
    weekday_activities=(
        ("company", "school"),
        ("leisure",),
        ("household",),
    ),
    weekend_activities=(("leisure",), ("household",)),
)

time1 = time()
with torch.no_grad():
    result = model(timer=timer, susceptibilities=susc)
time2 = time()
print(f"Took {time2-time1} seconds.")
