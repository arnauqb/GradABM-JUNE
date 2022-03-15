from torch_june import GraphLoader, InfectionPassing, AgentDataLoader, TorchJune, Timer
from torch_geometric.data import HeteroData
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import h5py
import networkx
from torch_june.june_world_loader import LeisureNetworkLoader
from torch_june.infections import Infections, InfectionSampler
from torch_june.utils import generate_erdos_renyi
from torch.distributions import Normal, LogNormal
from time import time

def process_infected(infected, timer):
    timer.reset()
    dates = []
    while timer.date < timer.final_date:
        dates.append(timer.date)
        next(timer)
    s_infected = infected.sum(1).cpu()
    df = pd.DataFrame(index=dates, data=s_infected, columns=["new_infected"])
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df.groupby(df.index.date).sum()


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"


betas = {"company": 0.5, "school": 0.5, "household": 0.5, "leisure": 0.5}

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
susc = torch.tensor(susc).to(device)

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

<<<<<<< HEAD
fig, ax = plt.subplots()
ax.plot(torch.sum(result, 1).cpu().detach(), "o-")
fig.savefig("./result.png")
=======
df = process_infected(infected=result, timer=timer)
fig, ax = plt.subplots()
df.plot(ax=ax, style="o-")
fig.autofmt_xdate()
fig.savefig("./results.pdf")
plt.show()
>>>>>>> f5ee6e695dd28d70daff70880720742ccee844a8
