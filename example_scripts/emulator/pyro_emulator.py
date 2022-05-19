import torch
import gpytorch
import pyro
import pandas as pd
import os, sys
from pathlib import Path

this_path = Path(os.path.abspath(__file__)).parent
sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_timer,
    get_data,
    get_people_by_age,
    backup_inf_data,
    run_model,
    fix_seed,
)
from parameters import make_parameters
from train_emulator import MultitaskGPModel, get_training_data
from torch_june import TorchJune, Policies

fix_seed()

device = "cuda:1"
DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ne.pkl"
TIMER = make_timer()
DATA = get_data(DATA_PATH, n_seed=2000, device=device)
n_agents = DATA["agent"]["id"].shape[0]
people_by_age = get_people_by_age(DATA, device)
BACKUP = backup_inf_data(DATA)
PARAMETERS = make_parameters()
time_stamps = [10, 30, 60, 90]
n_tasks = len(time_stamps)


true_log_beta_household = torch.tensor(-0.5, device=device)
true_log_beta_company = torch.tensor(0.0, device=device)
true_log_beta_school = torch.tensor(0.1, device=device)
true_log_beta_leisure = torch.tensor(-1.5, device=device)
true_log_beta_university = torch.tensor(-2.0, device=device)
true_log_beta_care_home = torch.tensor(-2.0, device=device)

with torch.no_grad():
    june_model = TorchJune(
        log_beta_household=true_log_beta_household,
        log_beta_school=true_log_beta_school,
        log_beta_company=true_log_beta_company,
        log_beta_university=true_log_beta_university,
        log_beta_leisure=true_log_beta_leisure,
        log_beta_care_home=true_log_beta_care_home,
        policies=Policies.from_parameters(PARAMETERS),
        device=device,
    )
    dates, cases, deaths, cases_by_age = run_model(
        model=june_model, timer=TIMER, data=DATA, backup=BACKUP
    )
    true_cases = cases / n_agents
    true_cases = torch.log10(true_cases[time_stamps])

# Emulator

train_x, train_y, val_x, val_y = get_training_data(device)
n_tasks = len(time_stamps)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks).to(
    device
)
model = MultitaskGPModel(train_x, train_y, likelihood, n_tasks=n_tasks).to(device)
state_dict = torch.load("./emulator_120.pth")
model.load_state_dict(state_dict)

model.eval()
likelihood.eval()


def pyro_model(true_data):
    log_beta_household = pyro.sample(
        "log_beta_household", pyro.distributions.Normal(-0.7, 0.1)
    ).to(device)
    log_beta_school = pyro.sample(
       "log_beta_school", pyro.distributions.Normal(-0.1, 0.1)
    ).to(device)
    # log_beta_company = pyro.sample(
    #    "log_beta_company", pyro.distributions.Uniform(-0.5, 0.5)
    # ).to(device)
    # log_beta_leisure = pyro.sample(
    #    "log_beta_leisure", pyro.distributions.Uniform(-1.7, -1.3)
    # ).to(device)
    with torch.no_grad():
        test_x = torch.tensor(
            [
                [
                    log_beta_household,
                    log_beta_school,
                    # log_beta_company,
                    # log_beta_leisure,
                ]
            ],
            device=device,
        )
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
    cases = mean.flatten()
    #error_emulator = torch.sqrt((mean - lower) ** 2 + (mean - upper) ** 2)
    error_emulator = (abs(mean - lower) + abs(mean - upper)) / 2
    error = error_emulator.flatten() + 0.025
    #print("------------")
    #print(f"cases {cases}")
    #print(f"true {true_data}")
    #print(f"error {error}")
    #print("------------\n")
    pyro.sample(
        "cases",
        pyro.distributions.Normal(cases, error),
        obs=true_data,
    )


def logger(kernel, samples, stage, i, dfs):
    df = dfs[stage]
    for key in samples:
        if "beta" not in key:
            continue
        unconstrained_samples = samples[key].detach()
        constrained_samples = kernel.transforms[key].inv(unconstrained_samples)
        df.loc[i, key] = constrained_samples.cpu().item()
    df.to_csv(f"./results_emulator_leisure_{stage}_lowtree.csv", index=False)


def run_mcmc(true_data):
    dfs = {"Sample": pd.DataFrame(), "Warmup": pd.DataFrame()}
    mcmc_kernel = pyro.infer.NUTS(pyro_model, max_tree_depth=6)
    mcmc = pyro.infer.MCMC(
        mcmc_kernel,
        num_samples=10000,
        warmup_steps=2000,
        hook_fn=lambda kernel, samples, stage, i: logger(
            kernel, samples, stage, i, dfs
        ),
    )
    mcmc.run(true_data)
    print(mcmc.summary())
    print(mcmc.diagnostics())


run_mcmc(true_cases)
