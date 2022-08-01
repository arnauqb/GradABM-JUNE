from torch_june.runner import Runner
import torch
import sys

with torch.no_grad():
    runner = Runner.from_file(sys.argv[1])
    results, is_infected = runner()

runner.save_results(results, is_infected)
