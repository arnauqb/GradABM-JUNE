import pytest
import yaml
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path

from grad_june.runner import Runner
from grad_june.paths import default_config_path


class TestRunner:
    @pytest.fixture(name="runner")
    def make_runner(self):
        with open(default_config_path, "r") as f:
            parameters = yaml.safe_load(f)
        runner = Runner.from_parameters(parameters)
        for key in runner.model.infection_networks.networks.keys():
            runner.model.infection_networks.networks[key].log_beta = torch.nn.Parameter(
                runner.model.infection_networks.networks[key].log_beta
            )
        return runner

    def test__read_from_file(self, runner):
        file_runner = Runner.from_file()
        assert file_runner.input_parameters == runner.input_parameters

    def test__get_data(self, runner):
        n_agents = runner.data["agent"].id.shape[0]
        inf_params = runner.data["agent"].infection_parameters
        shape = (3, n_agents)
        assert inf_params["max_infectiousness"].shape == shape
        assert inf_params["shape"].shape == shape
        assert inf_params["rate"].shape == shape
        assert inf_params["shift"].shape == shape

    def test__seed(self, runner):
        runner.set_initial_cases()
        assert np.isclose(
            runner.data["agent"].is_infected.sum().item(),
            0.10 * runner.n_agents,
            rtol=3e-1,
        )

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
        results = runner()
        n_timesteps = 31
        assert len(results["dates"]) == n_timesteps
        assert len(results["cases_per_timestep"]) == n_timesteps
        assert len(results["deaths_per_timestep"]) == n_timesteps
        assert len(results["cases_by_age_18"]) == n_timesteps
        assert len(results["cases_by_age_65"]) == n_timesteps
        assert len(results["cases_by_age_100"]) == n_timesteps

    def test__save_results(self, runner):
        with torch.no_grad():
            results = runner()
        runner.save_results(results)
        loaded_results = pd.read_csv("./example/results.csv", index_col=0)
        for key in results:
            if key in ("dates"):
                continue
            assert np.allclose(loaded_results[key], results[key].numpy())

    def test__deaths_gradient(self, runner):
        results = runner()
        assert results["cases_per_timestep"].requires_grad
        data = runner.data
        data_results = data["results"]
        daily_deaths = data_results["deaths_per_timestep"]
        assert (results["deaths_per_timestep"] == daily_deaths).all()
        assert daily_deaths.shape[0] == runner.input_parameters["timer"]["total_days"] + 1
        assert daily_deaths.requires_grad
