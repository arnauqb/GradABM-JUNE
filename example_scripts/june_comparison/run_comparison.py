import numpy as np
from pandas.core import groupby
import torch
import pickle
import pymultinest
from pathlib import Path
import pandas as pd
from time import time

this_path = Path(__file__).parent
import sys

sys.path.append(this_path.parent.as_posix())

from torch_june import TorchJune, Timer


def make_timer():
    return Timer(
        initial_day="2022-02-01",
        total_days=35,
        weekday_step_duration=(12, 12),
        # weekday_step_duration=(24,),
        weekend_step_duration=(24,),
        weekday_activities=(
            (
                "school",
                "university",
                "company",
                "care_home",
                # "leisure",
            ),
            (
                "care_home",
                "household",
            ),
        ),
        # weekday_activities=(("school",),),
        weekend_activities=(
            (
                "care_home",
                "household",
            ),
        ),
    )


def infector(data, indices):
    susc = data["agent"]["susceptibility"].cpu().numpy()
    is_inf = data["agent"]["is_infected"].cpu().numpy()
    inf_t = data["agent"]["infection_time"].cpu().numpy()
    next_stage = data["agent"]["symptoms"]["next_stage"].cpu().numpy()
    susc[indices] = 0.0
    is_inf[indices] = 1.0
    inf_t[indices] = 0.0
    next_stage[indices] = 2
    data["agent"]["susceptibility"] = torch.tensor(susc, device=device)
    data["agent"]["is_infected"] = torch.tensor(is_inf, device=device)
    data["agent"]["infection_time"] = torch.tensor(inf_t, device=device)
    data["agent"]["symptoms"]["next_stage"] = torch.tensor(next_stage, device=device)
    return data


def get_data(june_data_path, device, n_seed=1):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f).to(device)
    n_agents = len(data["agent"]["id"])
    # sampler = make_sampler()
    inf_params = {}
    # inf_params_values = sampler(n_agents)
    inf_params["max_infectiousness"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[0].to(device)
    inf_params["shape"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[1].to(device)
    inf_params["rate"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[2].to(device)
    inf_params["shift"] = torch.ones(
        n_agents, device=device
    )  # inf_params_values[3].to(device)
    data["agent"].infection_parameters = inf_params
    data["agent"].transmission = torch.zeros(n_agents, device=device)
    data["agent"].susceptibility = torch.ones(n_agents, device=device)
    data["agent"].is_infected = torch.zeros(n_agents, device=device)
    data["agent"].infection_time = torch.zeros(n_agents, device=device)
    symptoms = {}
    symptoms["current_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
    symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
    symptoms["time_to_next_stage"] = torch.zeros(n_agents, device=device)
    data["agent"].symptoms = symptoms
    data = infector(data, [0])
    return data


def backup_inf_data(data):
    ret = {}
    ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
    ret["is_infected"] = data["agent"].is_infected.detach().clone()
    ret["infection_time"] = data["agent"].infection_time.detach().clone()
    ret["transmission"] = data["agent"].transmission.detach().clone()
    symptoms = {}
    symptoms["current_stage"] = (
        data["agent"]["symptoms"]["current_stage"].detach().clone()
    )
    symptoms["next_stage"] = data["agent"]["symptoms"]["next_stage"].detach().clone()
    symptoms["time_to_next_stage"] = (
        data["agent"]["symptoms"]["time_to_next_stage"].detach().clone()
    )
    ret["symptoms"] = symptoms
    return ret


def restore_data(data, backup):
    data["agent"].transmission = backup["transmission"].detach().clone()
    data["agent"].susceptibility = backup["susceptibility"].detach().clone()
    data["agent"].is_infected = backup["is_infected"].detach().clone()
    data["agent"].infection_time = backup["infection_time"].detach().clone()
    data["agent"].symptoms["current_stage"] = (
        backup["symptoms"]["current_stage"].detach().clone()
    )
    data["agent"].symptoms["next_stage"] = (
        backup["symptoms"]["next_stage"].detach().clone()
    )
    data["agent"].symptoms["time_to_next_stage"] = (
        backup["symptoms"]["time_to_next_stage"].detach().clone()
    )
    return data


device = f"cuda:0"


def group_by_symptoms(symptoms, stages):
    current_stage = symptoms["current_stage"]
    ret = torch.zeros(len(stages), device=device)
    for i in range(len(stages)):
        this_stage = current_stage[current_stage == i]
        ret[i] = len(this_stage)
    return ret


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    # time_curve = torch.zeros(0, dtype=torch.float, device=device)
    time_curve = model(data, timer)["agent"].is_infected.sum()
    symptoms = group_by_symptoms(
        data["agent"].symptoms, model.symptoms_updater.symptoms_sampler.stages
    )
    # torch.zeros((0, ), dtype=torch.long, device=device)
    dates = [timer.date]
    while timer.date < timer.final_date:
        next(timer)
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        n_by_symptoms = group_by_symptoms(
            data["agent"].symptoms, model.symptoms_updater.symptoms_sampler.stages
        )
        symptoms = torch.vstack((symptoms, n_by_symptoms))
        dates.append(timer.date)
    return dates, time_curve, symptoms


def get_model_prediction(**kwargs):
    params = {"_".join(key.split("_")[1:]): 10 ** kwargs[key] for key in kwargs}
    model = TorchJune(**params, device=device)
    return run_model(model)


DATA_PATH = "./data.pkl"

DATA = get_data(DATA_PATH, device, n_seed=1)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

log_beta_household = torch.tensor(np.log10(1), device=device)
log_beta_school = torch.tensor(np.log10(5), device=device)
log_beta_company = torch.tensor(np.log10(1), device=device)
log_beta_leisure = torch.tensor(np.log10(5), device=device)
log_beta_university = torch.tensor(np.log10(1), device=device)
log_beta_care_home = torch.tensor(np.log10(1), device=device)

symptoms_cols = [
    "recovered",
    "susceptible",
    "exposed",
    "infectious",
    "symptomatic",
    "severe",
    "critical",
    "dead",
]

t1 = time()
dfs = []
for i in range(10):
    dates, true_data, true_symptoms = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_leisure=log_beta_leisure,
        log_beta_university=log_beta_university,
        log_beta_care_home=log_beta_care_home,
    )

    df = pd.DataFrame(index=dates)
    df.index.name = "time_stamp"
    cases = true_data.cpu().numpy()
    daily_cases = np.diff(cases, prepend=cases[0])
    df["infected"] = cases
    df["daily_infected"] = daily_cases
    true_symptoms = true_symptoms.cpu().numpy()
    df_symp = pd.DataFrame(index=dates, columns=symptoms_cols)
    for i, date in enumerate(dates):
        df_symp.loc[date] = true_symptoms[i, :]
    df = pd.merge(df, df_symp, left_index=True, right_index=True)
    dfs.append(df)
df = sum(dfs) / len(dfs)
df.to_csv("results.csv")

t2 = time()
print(f"Took {t2-t1:.2f} seconds.")
