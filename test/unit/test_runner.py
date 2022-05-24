import pytest
import yaml
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path

from torch_june.runner import Runner
from torch_june.paths import default_config_path


class TestRunner:
    @pytest.fixture(name="data_path")
    def get_data_path(self):
        data_path = Path(os.path.abspath(__file__)).parent.parent / "data/data.pkl"
        return data_path

    @pytest.fixture(name="runner")
    def make_runner(self, data_path):
        with open(default_config_path, "r") as f:
            parameters = yaml.safe_load(f)
        parameters["data_path"] = data_path
        return Runner.from_parameters(parameters)

    def test__read_from_file(self, runner, data_path):
        file_runner = Runner.from_file()
        # except data path all params should be equal
        file_runner.parameters["data_path"] = data_path
        assert file_runner.parameters == runner.parameters

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
        results = runner.run()
        assert len(results["dates"]) == 91
        assert len(results["cases_per_timestep"]) == 91
        assert len(results["deaths_per_timestep"]) == 91
        assert len(results["cases_by_age_18"]) == 91
        assert len(results["cases_by_age_25"]) == 91
        assert len(results["cases_by_age_65"]) == 91
        assert len(results["cases_by_age_80"]) == 91
        assert len(results["cases_by_age_100"]) == 91

    def test__save_results(self, runner):
        with torch.no_grad():
            results = runner.run()
        runner.save_results(results)
        loaded_results = pd.read_csv("./example/results.csv", index_col=0)
        for key in results:
            if key == "dates":
                continue
            assert (loaded_results[key] == results[key].numpy()).all()

