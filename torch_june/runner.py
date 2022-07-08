import torch
import pickle
import numpy as np
import pandas as pd
import yaml
import pyro
from pathlib import Path

from torch_june.paths import default_config_path
from torch_june import TorchJune, Timer, TransmissionSampler
from torch_june.utils import read_path
from torch_june.infection_seed import infect_people_at_indices


class Runner(pyro.nn.PyroModule):
    def __init__(
        self,
        model,
        data,
        timer,
        n_initial_cases,
        save_path,
        parameters,
        age_bins=(0, 18, 25, 65, 80, 100),
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.data_backup = self.backup_infection_data(data)
        self.timer = timer
        self.n_initial_cases = n_initial_cases
        self.device = model.device
        self.age_bins = torch.tensor(age_bins, device=self.device)
        self.n_agents = data["agent"].id.shape[0]
        self.population_by_age = self.get_people_by_age()
        self.save_path = Path(save_path)
        self.parameters = parameters

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        model = TorchJune.from_parameters(params)
        data = cls.get_data(params)
        timer = Timer.from_parameters(params)
        return cls(
            model=model,
            data=data,
            timer=timer,
            n_initial_cases=params["infection_seed"]["n_initial_cases"],
            save_path=params["save_path"],
            parameters=params,
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
        symptoms = {}
        symptoms["current_stage"] = torch.ones(
            n_agents, dtype=torch.long, device=device
        )
        symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
        symptoms["time_to_next_stage"] = torch.zeros(n_agents, device=device)
        data["agent"].symptoms = symptoms
        return data

    # def reset_model(self):
    #    self.model = TorchJune.from_parameters(self.parameters)

    def backup_infection_data(self, data):
        ret = {}
        ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
        ret["is_infected"] = data["agent"].is_infected.detach().clone()
        ret["infection_time"] = data["agent"].infection_time.detach().clone()
        ret["transmission"] = data["agent"].transmission.detach().clone()
        symptoms = {}
        symptoms["current_stage"] = (
            data["agent"]["symptoms"]["current_stage"].detach().clone()
        )
        symptoms["next_stage"] = (
            data["agent"]["symptoms"]["next_stage"].detach().clone()
        )
        symptoms["time_to_next_stage"] = (
            data["agent"]["symptoms"]["time_to_next_stage"].detach().clone()
        )
        ret["symptoms"] = symptoms
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
        self.data["agent"].symptoms["current_stage"] = (
            self.data_backup["symptoms"]["current_stage"].detach().clone()
        )
        self.data["agent"].symptoms["next_stage"] = (
            self.data_backup["symptoms"]["next_stage"].detach().clone()
        )
        self.data["agent"].symptoms["time_to_next_stage"] = (
            self.data_backup["symptoms"]["time_to_next_stage"].detach().clone()
        )

    def set_initial_cases(self):
        indices = np.arange(0, self.data["agent"].id.shape[0])
        np.random.shuffle(indices)
        indices = indices[: self.n_initial_cases]
        return infect_people_at_indices(self.data, indices, device=self.device)

    def forward(self):
        timer = self.timer
        model = self.model
        data = self.data
        timer.reset()
        self.restore_initial_data()
        self.set_initial_cases()
        # data = model(data, timer)
        cases_per_timestep = data["agent"].is_infected.sum()
        cases_by_age = self.get_cases_by_age(data)
        deaths_per_timestep = self.get_deaths_from_symptoms(data["agent"].symptoms)
        dates = [timer.date]
        i = 0
        while timer.date < timer.final_date:
            i += 1
            next(timer)
            data = model(data, timer)

            cases = data["agent"].is_infected.sum()
            cases_per_timestep = torch.hstack((cases_per_timestep, cases))
            deaths = self.get_deaths_from_symptoms(data["agent"].symptoms)
            deaths_per_timestep = torch.hstack((deaths_per_timestep, deaths))
            cases_age = self.get_cases_by_age(data)
            cases_by_age = torch.vstack((cases_by_age, cases_age))

            dates.append(timer.date)
        results = {
            "dates": dates,
            "cases_per_timestep": cases_per_timestep / self.n_agents,
            "daily_cases_per_timestep": torch.diff(
                cases_per_timestep, prepend=torch.tensor([0.0], device=self.device)
            )
            / self.n_agents,
            "deaths_per_timestep": deaths_per_timestep / self.n_agents,
        }
        for (i, key) in enumerate(self.age_bins[1:]):
            results[f"cases_by_age_{key:02d}"] = (
                cases_by_age[:, i] / self.population_by_age[i]
            )
        return results

    def save_results(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(index=results["dates"])
        df.index.name = "date"
        for key in results:
            if key == "dates":
                continue
            df[key] = results[key].detach().cpu().numpy()
        df.to_csv(self.save_path / "results.csv")

    def get_deaths_from_symptoms(self, symptoms):
        return torch.tensor(
            symptoms["current_stage"][symptoms["current_stage"] == 7].shape[0],
            device=self.device,
        )

    def get_cases_by_age(self, data):
        ret = torch.zeros(self.age_bins.shape[0] - 1, device=self.device)
        for i in range(1, self.age_bins.shape[0]):
            mask1 = data["agent"].age < self.age_bins[i]
            mask2 = data["agent"].age > self.age_bins[i - 1]
            mask = mask1 * mask2
            ret[i - 1] = ret[i - 1] + data["agent"].is_infected[mask].sum()
        return ret

    def get_people_by_age(self):
        ret = torch.zeros(self.age_bins.shape[0] - 1, device=self.device)
        for i in range(1, self.age_bins.shape[0]):
            mask1 = self.data["agent"].age < self.age_bins[i]
            mask2 = self.data["agent"].age > self.age_bins[i - 1]
            mask = mask1 * mask2
            ret[i - 1] = mask.sum()
        return ret
