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
        assert file_runner._parameters == runner._parameters

    def test__get_data(self, runner):
        n_agents = runner.data["agent"].id.shape
        inf_params = runner.data["agent"].infection_parameters
        assert inf_params["max_infectiousness"].shape == n_agents
        assert inf_params["shape"].shape == n_agents
        assert inf_params["rate"].shape == n_agents
        assert inf_params["shift"].shape == n_agents

    def test__seed(self, runner):
        runner.set_initial_cases()
        assert runner.data["agent"].is_infected.sum().item() == int(
            0.05 * runner.n_agents
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
        results, is_infected = runner()
        n_timesteps = 16
        assert len(results["dates"]) == n_timesteps
        assert len(results["cases_per_timestep"]) == n_timesteps
        assert len(results["deaths_per_timestep"]) == n_timesteps
        assert len(results["cases_by_age_18"]) == n_timesteps
        assert len(results["cases_by_age_25"]) == n_timesteps
        assert len(results["cases_by_age_65"]) == n_timesteps
        assert len(results["cases_by_age_80"]) == n_timesteps
        assert len(results["cases_by_age_100"]) == n_timesteps
        assert len(is_infected) == runner.n_agents

    def test__save_results(self, runner):
        with torch.no_grad():
            results, is_infected = runner()
        runner.save_results(results, is_infected)
        loaded_results = pd.read_csv("./example/results.csv", index_col=0)
        for key in results:
            if key in ("dates", "daily_deaths_by_district"):
                continue
            assert np.allclose(loaded_results[key], results[key].numpy())

    def test__deaths_gradient(self, runner):
        results, is_infected = runner()
        assert results["cases_per_timestep"].requires_grad
        data = runner.data
        data_results = data["results"]
        daily_deaths = data_results["daily_deaths"]
        assert (results["deaths_per_timestep"] == daily_deaths).all()
        assert daily_deaths.shape[0] == runner._parameters["timer"]["total_days"] + 1
        assert daily_deaths.requires_grad

    def test__save_deaths_by_district(self, runner):
        district_ids = torch.randint(0, 2, size=(runner.n_agents,))
        _, people_in_districts = torch.unique(district_ids, return_counts=True)
        data = runner.data
        symptoms = data["agent"].symptoms
        data["agent"].district = district_ids
        symptoms["current_stage"] = runner.model.symptoms_updater.stages_ids[
            -1
        ] * torch.ones(
            symptoms["current_stage"].shape
        )  # everyone is dead.
        runner.store_differentiable_deaths(data)
        deaths_by_district = data["results"]["daily_deaths_by_district"]
        assert set(deaths_by_district.numpy()) == set(people_in_districts.numpy())
