import numpy as np
import torch
import hamiltorch
from pathlib import Path

this_path = Path(__file__).parent
import sys

sys.path.append(this_path.parent.as_posix())
from script_utils import (
    make_sampler,
    get_data,
    backup_inf_data,
    restore_data,
    make_timer,
)

from torch_june import TorchJune

device = "cuda:0"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_model(model):
    timer.reset()
    data = restore_data(DATA, BACKUP)
    time_curve = torch.zeros(0, dtype=torch.float).to(device)
    while timer.date < timer.final_date:
        cases = model(data, timer)["agent"].is_infected.sum()
        time_curve = torch.hstack((time_curve, cases))
        next(timer)
    return time_curve


def get_model_prediction(beta_company, beta_household, beta_leisure, beta_school):
    # b1, b2, b3, b4 = betas
    # beta_dict = {
    #    "beta_company": b1,
    #    "beta_school": b2,
    #    "beta_household": b3,
    #    "beta_leisure": b4,
    # }
    #print("---")
    #print(beta_leisure)
    #print(beta_household)
    #print(beta_school)
    #print(beta_company)
    model = TorchJune(
        beta_leisure=beta_leisure,
        beta_household=beta_household,
        beta_school=beta_school,
        beta_company=beta_company,
    )
    return run_model(model)


def log_prob_func(params):
    beta_company, beta_school, beta_household, beta_leisure = params
    time_curve = get_model_prediction(
        beta_company=beta_company,
        beta_school=beta_school,
        beta_leisure=beta_leisure,
        beta_household=beta_household,
    )
    # time_curve = run_model(model)
    loglikelihood = (
        torch.distributions.Normal(
            time_curve, torch.ones(time_curve.shape[0], device=device)
        )
        .log_prob(true_data)
        .sum()
    )
    return loglikelihood


# DATA_PATH = "/cosma7/data/dp004/dc-quer1/data_ney.pkl"
DATA_PATH = "/home/arnau/code/torch_june/worlds/data_two_super_areas.pkl"

DATA = get_data(DATA_PATH, device, n_seed=100)
BACKUP = backup_inf_data(DATA)

timer = make_timer()

# log_params = torch.tensor([2.0, 3.0, 4.0, 1.0], requires_grad=True, device=device)
# true_params = torch.log10(log_params)
# b1, b2, b3, b4 = true_params
# beta_dict = {"company": b1, "school": b2, "household": b3, "leisure": b4}
beta_company = torch.tensor(-1.0, requires_grad=True, device=device)
beta_school = torch.tensor(1.0, requires_grad=True, device=device)
beta_household = torch.tensor(-1.0, requires_grad=True, device=device)
beta_leisure = torch.tensor(-1.0, requires_grad=True, device=device)

true_data = get_model_prediction(
    beta_company=beta_company,
    beta_household=beta_household,
    beta_school=beta_school,
    beta_leisure=beta_leisure,
)
# true_data.sum().backward()

#params = [beta_company, beta_school, beta_household, beta_leisure]
#ll = log_prob_func(params)
#ll.backward()
#
#print(beta_company.grad)
#print(beta_school.grad)
#print(beta_leisure.grad)
#print(beta_household.grad)

# true_model = TorchJune(parameters=beta_dict)
# true_data = run_model(true_model)
# true_data.sum().backward()


# log_likelihood = log_prob_func(true_model, true_data)
# log_likelihood.backward()
# print([p.grad for p in true_model.parameters()])

num_samples = 100
burn=10
step_size = 1e-3
num_steps_per_sample = 10
params_init = torch.tensor([-2.0, -1.0, 0.5, 1.0], device=device)

#
# beta_dict = {"company": -1.0, "school": 1.0, "household": -1.0, "leisure": 1.0}
#
#
params_hmc = hamiltorch.sample(
   log_prob_func=log_prob_func,
   params_init=params_init,
   num_samples=num_samples,
   step_size=step_size,
   num_steps_per_sample=num_steps_per_sample,
   burn=burn,
   sampler = hamiltorch.Sampler.HMC,
)
# print(params_hmc)
