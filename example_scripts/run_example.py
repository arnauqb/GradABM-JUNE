import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
from random import random
import torch
from torch.distributions import Normal, LogNormal

from torch_june import TorchJune, Timer, InfectionSampler


def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return InfectionSampler(max_infectiousness, shape, rate, shift)

def make_data(june_data_path, n_seed=1):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f)
    n_agents = len(data["agent"]["id"])
    sampler = make_sampler()
    inf_params = {}
    inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = inf_params_values[0]
    inf_params["shape"] = inf_params_values[1]
    inf_params["rate"] = inf_params_values[2]
    inf_params["shift"] = inf_params_values[3]
    data["agent"].infection_parameters = inf_params_values
    data["agent"].transmission = torch.zeros(n_agents)

    inf_choice = np.random.choice(range(len(data["agent"]["id"])), n_seed, replace=False)
    susceptibility = np.ones(n_agents)
    is_infected = np.zeros(n_agents)
    infection_time = -1.0 * np.ones(n_agents)
    susceptibility[inf_choice] = 0.0
    is_infected[inf_choice] = 1
    infection_time[inf_choice] = 0.0
    data["agent"].susceptibility = torch.tensor(susceptibility, dtype=torch.float)
    data["agent"].is_infected = torch.tensor(is_infected, dtype=torch.int)
    data["agent"].infection_time = torch.tensor(infection_time, dtype=torch.float)
    return data

def get_cases(data):
    return float(data["agent"].is_infected.detach().sum())

def get_cases_tensor(data):
    return data["agent"].is_infected.sum()

def get_daily_cases(active_cases, timer):
    daily_cases = np.diff(active_cases)
    timer.reset()
    dates = []
    while timer.date < timer.final_date:
        dates.append(timer.date)
        next(timer)
    df = pd.DataFrame(index=dates[:-1], data=daily_cases, columns=["daily_cases"])
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df.groupby(df.index.date).sum()
        
june_data_path = sys.argv[1]
data = make_data(june_data_path, 10).to("cuda:0")
data
