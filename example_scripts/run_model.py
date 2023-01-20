from grad_june.runner import Runner
import torch
import sys

runner = Runner.from_file(sys.argv[1])
runner.model.infection_networks.networks["household"].log_beta = torch.nn.Parameter(
    runner.model.infection_networks.networks["household"].log_beta
)
results, is_infected = runner()
cases = results["cases_per_timestep"].sum()
cases.backward()

#runner.save_results(results, is_infected)
