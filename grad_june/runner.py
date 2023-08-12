import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.checkpoint import checkpoint
import yaml
from pathlib import Path

from grad_june.paths import default_config_path
from grad_june import GradJune, Timer, TransmissionSampler
from grad_june.utils import read_path
from grad_june.demographics import (
    get_people_by_age,
    get_cases_by_age,
    store_differentiable_deaths,
)


class Runner(torch.nn.Module):
    def __init__(
        self,
        model,
        data,
        timer,
        save_path,
        parameters,
        age_bins=(0, 18, 65, 100),
        store_cases_by_age=False,
        store_differentiable_deaths=False,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.data_backup = self.backup_infection_data(data)
        self.timer = timer
        self.device = model.device
        self.age_bins = torch.tensor(age_bins, device=self.device)
        self.ethnicities = np.sort(np.unique(data["agent"].ethnicity))
        self.n_agents = data["agent"].id.shape[0]
        self.population_by_age = get_people_by_age(data["agent"].age, self.age_bins)
        self.save_path = Path(save_path)
        self.input_parameters = parameters
        self.restore_initial_data()
        self.store_cases_by_age = store_cases_by_age
        self.store_differentiable_deaths = store_differentiable_deaths

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        model = GradJune.from_parameters(params)
        data = cls.get_data(params)
        timer = Timer.from_parameters(params)
        age_bins_to_save = params.get("age_bins_to_save", (0, 18, 65, 100))
        store_differentiable_deaths = params.get("store_differentiable_deaths", False)
        store_cases_by_age = params.get("store_cases_by_age", False)
        return cls(
            model=model,
            data=data,
            timer=timer,
            save_path=params["save_path"],
            parameters=params,
            age_bins=age_bins_to_save,
            store_cases_by_age=store_cases_by_age,
            store_differentiable_deaths=store_differentiable_deaths,
        )

    @staticmethod
    def get_data(params):
        device = params["system"]["device"]
        data_path = read_path(params["data_path"])
        with open(data_path, "rb") as f:
            data = pickle.load(f).to(device)
        n_agents = len(data["agent"]["id"])
        inf_params = {}
        transmission_sampler = TransmissionSampler.from_parameters(params)
        transmission_values = transmission_sampler(n_agents)
        inf_params["max_infectiousness"] = transmission_values[0, :]
        inf_params["shape"] = transmission_values[1, :]
        inf_params["rate"] = transmission_values[2, :]
        inf_params["shift"] = transmission_values[3, :]
        data["agent"].infection_parameters = inf_params
        data["agent"].transmission = torch.zeros(n_agents, device=device)
        data["agent"].susceptibility = torch.ones(n_agents, device=device)
        data["agent"].is_infected = torch.zeros(n_agents, device=device)
        data["agent"].infection_time = torch.zeros(n_agents, device=device)
        #symptoms = {}
        #symptoms["current_stage"] = torch.ones(
        #    n_agents, dtype=torch.long, device=device
        #)
        #symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
        #symptoms["time_to_next_stage"] = torch.zeros(n_agents, device=device)
        #data["agent"].symptoms = symptoms
        return data

    def backup_infection_data(self, data):
        ret = {}
        ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
        ret["is_infected"] = data["agent"].is_infected.detach().clone()
        ret["infection_time"] = data["agent"].infection_time.detach().clone()
        ret["transmission"] = data["agent"].transmission.detach().clone()
        #symptoms = {}
        #symptoms["current_stage"] = (
        #    data["agent"]["symptoms"]["current_stage"].detach().clone()
        #)
        #symptoms["next_stage"] = (
        #    data["agent"]["symptoms"]["next_stage"].detach().clone()
        #)
        #symptoms["time_to_next_stage"] = (
        #    data["agent"]["symptoms"]["time_to_next_stage"].detach().clone()
        #)
        #ret["symptoms"] = symptoms
        return ret

    def restore_initial_data(self):
        self.data["agent"].transmission = (
            self.data_backup["transmission"].detach().clone()
        )
        self.data["agent"].susceptibility = (
            self.data_backup["susceptibility"].detach().clone()
        )
        self.data["agent"].is_infected = (
            self.data_backup["is_infected"].detach().clone()
        )
        self.data["agent"].infection_time = (
            self.data_backup["infection_time"].detach().clone()
        )
        #self.data["agent"].symptoms["current_stage"] = (
        #    self.data_backup["symptoms"]["current_stage"].detach().clone()
        #)
        #self.data["agent"].symptoms["next_stage"] = (
        #    self.data_backup["symptoms"]["next_stage"].detach().clone()
        #)
        #self.data["agent"].symptoms["time_to_next_stage"] = (
        #    self.data_backup["symptoms"]["time_to_next_stage"].detach().clone()
        #)
        # reset results
        self.data["results"] = {}
        self.data["results"]["deaths_per_timestep"] = None

    def forward(self):
        timer = self.timer
        model = self.model
        data = self.data
        timer.reset()
        self.restore_initial_data()
        model(data, timer)
        cases_per_timestep = data["agent"].is_infected.sum()
        if self.store_cases_by_age:
            cases_by_age = get_cases_by_age(data, self.age_bins)
        if self.store_differentiable_deaths:
            store_differentiable_deaths(
                data, self.model.symptoms_updater.stages_ids[-1]
            )
        dates = [timer.date]
        i = 0
        while timer.date < timer.final_date:
            i += 1
            next(timer)
            model(data, timer)
            cases = data["agent"].is_infected.sum()
            cases_per_timestep = torch.hstack((cases_per_timestep, cases))
            if self.store_cases_by_age:
                cases_age = get_cases_by_age(data, self.age_bins)
                cases_by_age = torch.vstack((cases_by_age, cases_age))
            if self.store_differentiable_deaths:
                store_differentiable_deaths(
                    data, self.model.symptoms_updater.stages_ids[-1]
                )
            dates.append(timer.date)
        results = {
            "dates": dates,
            "cases_per_timestep": cases_per_timestep,
            "daily_cases_per_timestep": torch.diff(
                cases_per_timestep, prepend=torch.tensor([0.0], device=self.device)
            ),
        }
        if self.store_cases_by_age:
            for i, key in enumerate(self.age_bins[1:]):
                results[f"cases_by_age_{key:02d}"] = cases_by_age[:, i]
        if self.store_differentiable_deaths:
            results["deaths_per_timestep"] = data["results"]["deaths_per_timestep"]
        return results

    def save_results(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(index=results["dates"])
        df.index.name = "date"
        for key in results:
            if key in ("dates"):
                continue
            df[key] = results[key].detach().cpu().numpy()
        df.to_csv(self.save_path / "results.csv")
