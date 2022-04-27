import torch
import os
import gc
import pandas as pd
from pyro import distributions
from pathlib import Path
import matplotlib.pyplot as plt

this_path = Path(os.path.abspath(__file__)).parent
import sys

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
    fix_seed,
    get_deaths_from_symptoms,
    get_cases_by_age,
    get_people_by_age,
)

from torch_june import TorchJune

# torch.autograd.set_detect_anomaly(True)
fix_seed()

device = "cuda:1"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DATA_PATH = "/home/arnau/code/torch_june/worlds/data_london.pkl"
DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
DATA = get_data(DATA_PATH, n_seed=100, device=device)
BACKUP = backup_inf_data(DATA)
TIMER = make_timer()


def run_model(model):
    TIMER.reset()
    data = restore_data(DATA, BACKUP)
    data = model(data, TIMER)
    time_curve = data["agent"].is_infected.sum()
    cases_by_age = get_cases_by_age(data, device=device)
    deaths_curve = get_deaths_from_symptoms(data["agent"].symptoms, device=device)
    dates = [TIMER.date]
    while TIMER.date < TIMER.final_date:
        next(TIMER)
        data = model(data, TIMER)

        cases = data["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        deaths = get_deaths_from_symptoms(data["agent"].symptoms, device=device)
        deaths_curve = torch.hstack((deaths_curve, deaths))
        cases_age = get_cases_by_age(data, device=device)
        cases_by_age = torch.vstack((cases_by_age, cases_age))

        dates.append(TIMER.date)
    return dates, time_curve, deaths_curve, cases_by_age


def get_model_prediction(**kwargs):
    input = {f"log_{key}": torch.log10(value) for key, value in kwargs.items()}
    model = TorchJune(**input, device=device)
    ret = run_model(model)
    return ret


def get_true_data(
    beta_household,
    beta_school,
    beta_company,
    beta_leisure,
    beta_university,
    beta_care_home,
    n=100,
):

    dates, true_cases, true_deaths, true_cases_by_age = get_model_prediction(
        beta_company=beta_company,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_care_home=beta_care_home,
        beta_university=beta_university,
    )
    true_cases = true_cases.reshape((1, *true_cases.shape))
    true_deaths = true_deaths.reshape((1, *true_deaths.shape))
    true_cases_by_age = true_cases_by_age.reshape((1, *true_cases_by_age.shape))
    for i in range(n - 1):

        _, true_cases2, true_deaths2, true_cases_by_age2 = get_model_prediction(
            beta_company=beta_company,
            beta_household=beta_household,
            beta_school=beta_school,
            beta_leisure=beta_leisure,
            beta_care_home=beta_care_home,
            beta_university=beta_university,
        )
        true_cases2 = true_cases2.reshape((1, *true_cases2.shape))
        true_deaths2 = true_deaths2.reshape((1, *true_deaths2.shape))
        true_cases_by_age2 = true_cases_by_age2.reshape((1, *true_cases_by_age2.shape))
        true_cases = torch.vstack((true_cases, true_cases2))
        true_deaths = torch.vstack((true_deaths, true_deaths2))
        true_cases_by_age = torch.vstack((true_cases_by_age, true_cases_by_age2))
    true_cases_mean = true_cases.mean(0)
    true_cases_std = true_cases.std(0)
    true_deaths_mean = true_deaths.to(torch.float).mean(0)
    true_deaths_std = true_deaths.to(torch.float).std(0)
    true_cases_by_age_mean = true_cases_by_age.mean(0)
    true_cases_by_age_std = true_cases_by_age.std(0)
    return (
        dates,
        true_cases_mean,
        true_cases_std,
        true_deaths_mean,
        true_deaths_std,
        true_cases_by_age_mean,
        true_cases_by_age_std,
    )


def log_likelihood(cases_by_age, true_cases_by_age, people_by_age, time_stamps):
    cases_by_age = cases_by_age[time_stamps, :] / people_by_age
    true_cases_by_age = true_cases_by_age[time_stamps, :] / people_by_age
    ll = (
        distributions.Normal(cases_by_age, 0.25 * cases_by_age + 1e-4)
        .log_prob(true_cases_by_age)
        .sum()
    )
    return -ll


def train_model(
    true_beta_household=0.3,
    true_beta_care_home=0.3,
    true_beta_company=0.3,
    true_beta_school=0.3,
    true_beta_leisure=0.3,
    true_beta_university=0.3,
    n_epochs=100,
):
    with torch.no_grad():
        (
            dates,
            true_cases_mean,
            true_cases_std,
            true_deaths_mean,
            true_deaths_std,
            true_cases_by_age_mean,
            true_cases_by_age_std,
        ) = get_true_data(
            beta_household=torch.tensor(true_beta_household),
            beta_school=torch.tensor(true_beta_school),
            beta_company=torch.tensor(true_beta_company),
            beta_leisure=torch.tensor(true_beta_leisure),
            beta_care_home=torch.tensor(true_beta_care_home),
            beta_university=torch.tensor(true_beta_university),
            n=1,
        )
    # fig, ax = plt.subplots()
    # ax.plot(dates, true_cases_mean.detach().cpu().numpy())
    # plt.show()

    # loss_fn = torch.nn.MSELoss(reduction="mean")
    loss_fn = torch.nn.L1Loss(reduction="mean")
    model = TorchJune(device=device)
    model.infection_passing.log_beta_care_home = (
        model.infection_passing.log_beta_household
    )
    model.infection_passing.log_beta_leisure = (
        model.infection_passing.log_beta_household
    )
    model.infection_passing.log_beta_school = model.infection_passing.log_beta_household
    model.infection_passing.log_beta_university = (
        model.infection_passing.log_beta_household
    )
    model.infection_passing.log_beta_company = (
        model.infection_passing.log_beta_household
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer, step_size=100, verbose=True, gamma=0.9
    # )
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_agents = DATA["agent"]["id"].shape[0]
    df = pd.DataFrame()
    people_by_age = get_people_by_age(DATA, device)
    time_stamps = [5, 10, 15, 20, 25]

    for epoch in range(n_epochs):
        # get the inputs; data is a list of [inputs, labels]
        data = restore_data(DATA, BACKUP)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        dates, cases, deaths_curve, cases_by_age = run_model(model)
        # loss = loss_fn(cases[-1] / n_agents, true_cases_mean[-1] / n_agents)
        # loss += 100 * loss_fn(
        #   deaths_curve[-1] / n_agents, true_deaths_mean[-1] / n_agents
        # )
        # loss = loss_fn(
        #   cases_by_age[time_stamps, :] / people_by_age,
        #   true_cases_by_age_mean[time_stamps, :] / people_by_age,
        # )
        loss = log_likelihood(
            cases_by_age=cases_by_age,
            true_cases_by_age=true_cases_by_age_mean,
            people_by_age=people_by_age,
            time_stamps=time_stamps,
        )
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        # print statistics
        running_loss = loss.item()
        print(f"[{epoch + 1}] loss: {running_loss:.10e}")
        for param in model.named_parameters():
            df.loc[epoch, param[0].split(".")[-1]] = param[1].item()
        df.loc[epoch, "loss"] = running_loss
        df.to_csv("gd_results_likelihood.csv", index=False)
    PATH = "./model.pth"
    torch.save(model.state_dict(), PATH)


train_model(n_epochs=50000)
