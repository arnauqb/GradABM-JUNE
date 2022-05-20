import pytest
import yaml
import torch
import os
from pathlib import Path

from torch_june.runner import Runner
from torch_june.paths import default_config_path


class TestRunner:
    @pytest.fixture(name="runner")
    def make_runner(self):
        data_path = Path(os.path.abspath(__file__)).parent.parent / "data/data.pkl"
        with open(default_config_path, "r") as f:
            parameters = yaml.safe_load(f)
        parameters["data_path"] = data_path
        return Runner.from_parameters(parameters)

    def test__get_data(self, runner):
        n_agents = runner.data["agent"].id.shape
        inf_params = runner.data["agent"].infection_parameters
        assert inf_params["max_infectiousness"].shape == n_agents
        assert inf_params["shape"].shape == n_agents
        assert inf_params["rate"].shape == n_agents
        assert inf_params["shift"].shape == n_agents

    def test__seed(self, runner):
        runner.set_initial_cases()
        assert runner.data["agent"].is_infected.sum().item() == 200

    def test__restore_data(self, runner):
        n_agents = runner.data["agent"].id.shape
        runner.data["agent"].transmission = torch.rand(n_agents)
        runner.data["agent"].susceptibility = torch.rand(n_agents)
        runner.data["agent"].is_infected = torch.rand(n_agents)
        runner.data["agent"].infection_time = torch.rand(n_agents)
        runner.data["agent"].symptoms["current_stage"] = torch.rand(n_agents)
        runner.data["agent"].symptoms["next_stage"] = torch.rand(n_agents)
        runner.data["agent"].symptoms["time_to_next_stage"] = torch.rand(n_agents)
        runner.restore_initial_data()
        assert (
            runner.data["agent"].symptoms["current_stage"].sum().item() == n_agents[0]
        )  # everyone is susceptible
        assert runner.data["agent"].symptoms["next_stage"].sum().item() == n_agents[0]
        assert runner.data["agent"].symptoms["time_to_next_stage"].sum().item() == 0

    def test__run_model(self, runner):
        runner.run()
        results = runner.results
        assert len(results["dates"]) == 91
        assert len(results["cases_per_timestep"]) == 91
        assert len(results["deaths_per_timestep"]) == 91
        assert results["cases_by_age"].shape == torch.Size([91, 4])
