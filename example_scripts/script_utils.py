import torch
import numpy as np
import pickle
import random
from pyro.distributions import Normal, LogNormal

from torch_june import TransmissionSampler, Timer, TorchJune
from torch_june.policies import Policies


def fix_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Fixing seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return TransmissionSampler(max_infectiousness, shape, rate, shift)


def infector(data, indices, device):
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


def get_data(june_data_path, device, n_seed=10):
    with open(june_data_path, "rb") as f:
        data = pickle.load(f).to(device)
    n_agents = len(data["agent"]["id"])
    inf_params = {}
    inf_params["max_infectiousness"] = torch.ones(
        n_agents, device=device
    )  
    inf_params["shape"] = torch.ones(
        n_agents, device=device
    ) 
    inf_params["rate"] = torch.ones(
        n_agents, device=device
    )
    inf_params["shift"] = torch.ones(
        n_agents, device=device
    )
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
    indices = np.arange(0, n_agents)
    # np.random.shuffle(indices)
    indices = indices[:n_seed]
    data = infector(data=data, indices=indices, device=device)
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


def make_timer():
    return Timer(
        initial_day="2022-02-01",
        total_days=90,
        weekday_step_duration=(24,),
        weekend_step_duration=(24,),
        weekday_activities=(
            ("company", "school", "university", "leisure", "care_home", "household"),
        ),
        weekend_activities=(("leisure", "care_home", "household"),),
    )


def group_by_symptoms(symptoms, stages, device):
    current_stage = symptoms["current_stage"]
    ret = torch.zeros(len(stages), device=device)
    for i in range(len(stages)):
        this_stage = current_stage[current_stage == i]
        ret[i] = len(this_stage)
    return ret


def get_deaths_from_symptoms(symptoms, device):
    return torch.tensor(
        symptoms["current_stage"][symptoms["current_stage"] == 7].shape[0],
        device=device,
    )


def get_cases_by_age(data, device):
    ages = torch.tensor([0, 18, 25, 65, 80], device=device)
    ret = torch.zeros(ages.shape[0] - 1, device=device)
    for i in range(1, ages.shape[0]):
        mask1 = data["agent"].age < ages[i]
        mask2 = data["agent"].age > ages[i - 1]
        mask = mask1 * mask2
        ret[i - 1] = ret[i - 1] + data["agent"].is_infected[mask].sum()
    return ret


def get_people_by_age(data, device):
    ages = torch.tensor([0, 18, 25, 65, 80], device=device)
    ret = torch.zeros(ages.shape[0] - 1, device=device)
    for i in range(1, ages.shape[0]):
        mask1 = data["agent"].age < ages[i]
        mask2 = data["agent"].age > ages[i - 1]
        mask = mask1 * mask2
        ret[i - 1] = mask.sum()
    return ret


def run_model(model, timer, data, backup):
    # print("----")
    device=model.device
    timer.reset()
    data = restore_data(data, backup)
    data = model(data, timer)
    time_curve = data["agent"].is_infected.sum()
    cases_by_age = get_cases_by_age(data, device=device)
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms, device=device)
    dates = [timer.date]
    i = 0
    while timer.date < timer.final_date:
        i += 1
        next(timer)
        data = model(data, timer)

        cases = data["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms, device=device)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        cases_age = get_cases_by_age(data, device=device)
        cases_by_age = torch.vstack((cases_by_age, cases_age))

        dates.append(timer.date)
    return np.array(dates), time_curve, deaths_curve, cases_by_age


def get_model_prediction(timer, data, backup, **kwargs):
    model = TorchJune(**kwargs, policies = Policies([]))
    ret = run_model(model=model, timer=timer, data=data, backup=backup)
    return ret


def get_average_predictions(
    log_beta_household,
    log_beta_school,
    log_beta_company,
    log_beta_leisure,
    log_beta_university,
    log_beta_care_home,
    timer,
    data,
    backup,
    device,
    n=1,
):

    dates, cases, deaths, cases_by_age = get_model_prediction(
        log_beta_company=log_beta_company,
        log_beta_household=log_beta_household,
        log_beta_school=log_beta_school,
        log_beta_leisure=log_beta_leisure,
        log_beta_care_home=log_beta_care_home,
        log_beta_university=log_beta_university,
        timer=timer,
        data=data,
        backup=backup,
        device=device
    )
    cases = cases.reshape((1, *cases.shape))
    deaths = deaths.reshape((1, *deaths.shape))
    cases_by_age = cases_by_age.reshape((1, *cases_by_age.shape))
    for i in range(n - 1):
        _, cases2, deaths2, cases_by_age2 = get_model_prediction(
            log_beta_company=log_beta_company,
            log_beta_household=log_beta_household,
            log_beta_school=log_beta_school,
            log_beta_leisure=log_beta_leisure,
            log_beta_care_home=log_beta_care_home,
            log_beta_university=log_beta_university,
            timer=timer,
            data=data,
            device=device,
            backup=backup,
        )
        cases2 = cases2.reshape((1, *cases2.shape))
        deaths2 = deaths2.reshape((1, *deaths2.shape))
        cases_by_age2 = cases_by_age2.reshape((1, *cases_by_age2.shape))
        cases = torch.vstack((cases, cases2))
        deaths = torch.vstack((deaths, deaths2))
        cases_by_age = torch.vstack((cases_by_age, cases_by_age2))
    cases_mean = cases.mean(0)
    cases_std = cases.std(0)
    deaths_mean = deaths.to(torch.float).mean(0)
    deaths_std = deaths.to(torch.float).std(0)
    cases_by_age_mean = cases_by_age.mean(0)
    cases_by_age_std = cases_by_age.std(0)
    return (
        dates,
        cases_mean,
        cases_std,
        deaths_mean,
        deaths_std,
        cases_by_age_mean,
        cases_by_age_std,
    )
