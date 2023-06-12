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
from grad_june.infection import infect_fraction_of_people
from grad_june.vaccination import Vaccines


class Runner(torch.nn.Module):
    def __init__(
        self,
        model,
        data,
        timer,
        log_fraction_initial_cases,
        save_path,
        parameters,
        age_bins=(0, 18, 65, 100),
        vaccines=None,
        initial_cases_infection_type=0,
    ):
        """
        Runner class to facilitate running the model.

        **Arguments**

        - `model`: The GradJune model to run.
        - `data`: The data to run the model on.
        - `timer`: The timer to use for the model.
        - `log_fraction_initial_cases`: The log of the fraction of people to infect
        - `save_path`: The path to save the results to.
        - `parameters`: The parameters used to run the model.
        - `age_bins`: The age bins that are used to save the cases / deaths results.
        - `vaccines`: The vaccines to use in the model.
        - `initial_cases_infection_type`: The infection type to use for the initial number of cases.
        """
        super().__init__()
        self.model = model
        self.data = data
        self.data_backup = self.backup_infection_data(data)
        self.timer = timer
        self.log_fraction_initial_cases = log_fraction_initial_cases
        self.initial_cases_infection_type = initial_cases_infection_type
        self.device = model.device
        self.age_bins = torch.tensor(age_bins, device=self.device)
        self.ethnicities = np.sort(np.unique(data["agent"].ethnicity))
        self.n_agents = data["agent"].id.shape[0]
        self.population_by_age = self.get_people_by_age()
        self.save_path = Path(save_path)
        self.input_parameters = parameters
        self.vaccines = vaccines
        self.restore_initial_data()

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
        if "vaccines" in params:
            vaccines = Vaccines.from_parameters(params["vaccines"], device=model.device)
        else:
            vaccines = None
        return cls(
            model=model,
            data=data,
            timer=timer,
            log_fraction_initial_cases=params["infection_seed"][
                "log_fraction_initial_cases"
            ],
            save_path=params["save_path"],
            vaccines=vaccines,
            parameters=params,
            age_bins = age_bins_to_save
        )

    @staticmethod
    def get_data(params):
        device = params["system"]["device"]
        data_path = read_path(params["data_path"])
        with open(data_path, "rb") as f:
            data = pickle.load(f).to(device)
        n_agents = len(data["agent"]["id"])
        transmission_sampler = TransmissionSampler.from_parameters(params)
        parameters_per_infection = transmission_sampler(n_agents)
        n_infections = parameters_per_infection["n_infections"]
        data["agent"].infection_parameters = parameters_per_infection
        data["agent"].transmission = torch.zeros(n_agents, device=device)
        data["agent"].susceptibility = torch.ones(
            (n_infections, n_agents), device=device
        )
        data["agent"].symptoms_susceptibility = torch.ones(
            (n_infections, n_agents), device=device
        )
        data["agent"].is_infected = torch.zeros(n_agents, device=device)
        data["agent"].infection_time = torch.zeros(n_agents, device=device)
        data["agent"].infection_id = torch.zeros(n_agents, dtype=torch.long)
        symptoms = {}
        symptoms["current_stage"] = torch.ones(
            n_agents, dtype=torch.long, device=device
        )
        symptoms["next_stage"] = torch.ones(n_agents, dtype=torch.long, device=device)
        symptoms["time_to_next_stage"] = torch.zeros(n_agents, device=device)
        data["agent"].symptoms = symptoms
        return data

    def backup_infection_data(self, data):
        ret = {}
        ret["susceptibility"] = data["agent"].susceptibility.detach().clone()
        ret["symptoms_susceptibility"] = (
            data["agent"].symptoms_susceptibility.detach().clone()
        )
        ret["is_infected"] = data["agent"].is_infected.detach().clone()
        ret["infection_id"] = data["agent"].infection_id.detach().clone()
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
        self.data["agent"].symptoms_susceptibility = (
            self.data_backup["symptoms_susceptibility"].detach().clone()
        )
        self.data["agent"].is_infected = (
            self.data_backup["is_infected"].detach().clone()
        )
        self.data["agent"].infection_id = (
            self.data_backup["infection_id"].detach().clone()
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
        # reset results
        self.data["results"] = {}
        self.data["results"]["deaths_per_timestep"] = None

    def set_initial_cases(self):
        fraction_initial_cases = 10.0**self.log_fraction_initial_cases
        new_infected = infect_fraction_of_people(
            data=self.data,
            timer=self.timer,
            symptoms_updater=self.model.symptoms_updater,
            device=self.device,
            fraction=fraction_initial_cases,
            infection_type=self.initial_cases_infection_type,
        )
        self.model.symptoms_updater(
            data=self.data, timer=self.timer, new_infected=new_infected
        )

    def vaccinate(self):
        if self.vaccines:
            susc, symp_susc = self.vaccines.vaccinate(
                ages=self.data["agent"].age,
                susceptibilities=self.data["agent"].susceptibility,
                symptom_susceptibilities=self.data["agent"].symptoms_susceptibility,
            )
            self.data["agent"].susceptibility = susc
            self.data["agent"].symptoms_susceptibility = symp_susc

    def forward(self):
        timer = self.timer
        model = self.model
        data = self.data
        timer.reset()
        self.restore_initial_data()
        self.set_initial_cases()
        self.vaccinate()
        # initialize results
        cases_per_timestep = data["agent"].is_infected.sum()
        cases_by_age = self.get_cases_by_age(data)
        self.store_differentiable_deaths(data)
        dates = [timer.date]
        i = 0
        while timer.date < timer.final_date:
            i += 1
            next(timer)
            data = model(data, timer)
            cases = data["agent"].is_infected.sum()
            cases_per_timestep = torch.hstack((cases_per_timestep, cases))
            self.store_differentiable_deaths(data)
            cases_age = self.get_cases_by_age(data)
            cases_by_age = torch.vstack((cases_by_age, cases_age))
            dates.append(timer.date)
        results = {
            "dates": dates,
            "cases_per_timestep": cases_per_timestep,
            "daily_cases_per_timestep": torch.diff(
                cases_per_timestep, prepend=torch.tensor([0.0], device=self.device)
            ),
            "deaths_per_timestep": data.results["deaths_per_timestep"],
        }
        for (i, key) in enumerate(self.age_bins[1:]):
            results[f"cases_by_age_{key:02d}"] = cases_by_age[:, i]
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

    def store_differentiable_deaths(self, data):
        """
        Returns differentiable deaths. The results are stored
        in data["results"]
        """
        symptoms = data["agent"].symptoms
        dead_idx = self.model.symptoms_updater.stages_ids[-1]
        deaths = (
            (symptoms["current_stage"] == dead_idx)
            * symptoms["current_stage"]
            / dead_idx
        )
        if data["results"]["deaths_per_timestep"] is not None:
            data["results"]["deaths_per_timestep"] = torch.hstack(
                (data["results"]["deaths_per_timestep"], deaths.sum())
            )
        else:
            data["results"]["deaths_per_timestep"] = deaths.sum()

    def get_cases_by_age(self, data):
        ret = torch.zeros(self.age_bins.shape[0] - 1, device=self.device)
        for i in range(1, self.age_bins.shape[0]):
            mask1 = data["agent"].age < self.age_bins[i]
            mask2 = data["agent"].age > self.age_bins[i - 1]
            mask = mask1 * mask2
            ret[i - 1] = (data["agent"].is_infected * mask).sum()
        return ret

    def get_people_by_age(self):
        ret = {}
        for i in range(1, self.age_bins.shape[0]):
            mask1 = self.data["agent"].age < self.age_bins[i]
            mask2 = self.data["agent"].age > self.age_bins[i - 1]
            mask = mask1 * mask2
            ret[int(self.age_bins[i].item())] = mask.sum()
        return ret

    def get_cases_by_ethnicity(self, data):
        ret = torch.zeros(len(self.ethnicities), device=self.device)
        for i, ethnicity in enumerate(self.ethnicities):
            mask = torch.tensor(
                self.data["agent"].ethnicity == ethnicity, device=self.device
            )
            ret[i] = (mask * data["agent"].is_infected).sum()
        return ret
