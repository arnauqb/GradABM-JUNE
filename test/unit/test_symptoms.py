from tkinter import W
from pytest import fixture
import numpy as np
import pyro
import torch

from grad_june.symptoms import SymptomsSampler, SymptomsUpdater
from grad_june.default_parameters import make_parameters


class TestSymptomsSampler:
    @fixture(name="input")
    def test__input(self):
        input = {
            "stages": [
                "recovered",
                "susceptible",
                "asymptomatic",
                "symptomatic",
                "critical",
                "dead",
            ],
            "stage_transition_probabilities": {
                "asymptomatic": {"0-50": 1.0, "50-100": 0.5},
                "symptomatic": {"0-100": 0.2},
                "critical": {"0-100": 0.1},
            },
            "stage_transition_times": {
                "asymptomatic": {"dist": "LogNormal", "loc": 1.1, "scale": 0.2},
                "symptomatic": {"dist": "Normal", "loc": 10.2, "scale": 3.0},
                "critical": {"dist": "LogNormal", "loc": 1.7, "scale": 0.5},
            },
            "recovery_times": {
                "asymptomatic": {"dist": "LogNormal", "loc": 1.2, "scale": 0.3},
                "symptomatic": {"dist": "LogNormal", "loc": 1.3, "scale": 0.5},
                "critical": {"dist": "LogNormal", "loc": 1.4, "scale": 0.8},
            },
        }
        return input

    @fixture(name="sp")
    def make_sp(self, input):
        params = {"system": {"device": "cpu"}, "symptoms": input}
        return SymptomsSampler.from_parameters(params)

    def test__read_input(self, sp):
        assert sp.stages == [
            "recovered",
            "susceptible",
            "asymptomatic",
            "symptomatic",
            "critical",
            "dead",
        ]
        assert (sp.stage_transition_probabilities[0] == torch.zeros(100)).all()
        assert (sp.stage_transition_probabilities[1] == torch.zeros(100)).all()
        assert (sp.stage_transition_probabilities[2][:50] == 1.0 * torch.ones(50)).all()
        assert (sp.stage_transition_probabilities[2][50:] == 0.5 * torch.ones(50)).all()
        assert (sp.stage_transition_probabilities[3] == 0.2 * torch.ones(100)).all()
        assert (sp.stage_transition_probabilities[4] == 0.1 * torch.ones(100)).all()

        assert sp.stage_transition_times[2].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.stage_transition_times[2].loc.item(), 1.1)
        assert np.isclose(sp.stage_transition_times[2].scale.item(), 0.2)
        assert np.isclose(
            sp.stage_transition_times[2].mean.item(), np.exp(1.1 + 0.2**2 / 2)
        )
        assert sp.stage_transition_times[3].__class__.__name__ == "Normal"
        assert np.isclose(sp.stage_transition_times[3].loc.item(), 10.2)
        assert np.isclose(sp.stage_transition_times[3].scale.item(), 3.0)
        assert np.isclose(sp.stage_transition_times[3].mean.item(), 10.2)
        assert sp.stage_transition_times[4].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.stage_transition_times[4].loc.item(), 1.7)
        assert np.isclose(sp.stage_transition_times[4].scale.item(), 0.5)
        assert np.isclose(
            sp.stage_transition_times[4].mean.item(), np.exp(1.7 + 0.5**2 / 2)
        )

        assert sp.recovery_times[2].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.recovery_times[2].loc.item(), 1.2)
        assert np.isclose(sp.recovery_times[2].scale.item(), 0.3)
        assert np.isclose(sp.recovery_times[2].mean.item(), np.exp(1.2 + 0.3**2 / 2))
        assert sp.recovery_times[3].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.recovery_times[3].loc.item(), 1.3)
        assert np.isclose(sp.recovery_times[3].scale.item(), 0.5)
        assert np.isclose(sp.recovery_times[3].mean.item(), np.exp(1.3 + 0.5**2 / 2))
        assert sp.recovery_times[4].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.recovery_times[4].loc.item(), 1.4)
        assert np.isclose(sp.recovery_times[4].scale.item(), 0.8)
        assert np.isclose(sp.recovery_times[4].mean.item(), np.exp(1.4 + 0.8**2 / 2))

    def test__sample(self, sp):
        ages = torch.tensor([0, 20, 40, 60, 80, 99])
        current_stage = torch.tensor([0, 1, 2, 3, 4, 5])
        next_stage = torch.tensor([0, 1, 3, 4, 5, 5])
        time_to_next_stage = torch.tensor([1.1, 2.5, 0.7, 0.9, 0.1, 0.5])
        symptoms_susceptibility = torch.ones((3, 6))
        infection_ids = torch.zeros(6, dtype=torch.long)

        time = 1.0
        need_to_trans = sp._get_need_to_transition(
            current_stage, time_to_next_stage, time
        )
        assert (need_to_trans == torch.tensor([0, 0, 1, 1, 1, 0]).to(torch.bool)).all()
        probability_next_symptomatic_stage = sp._get_prob_next_symptoms_stage(
            ages,
            next_stage,
            symptoms_susceptibility=symptoms_susceptibility,
            infection_ids=infection_ids,
        )
        assert (
            probability_next_symptomatic_stage
            == torch.tensor([0.0, 0.0, 0.2, 0.1, 0.0, 0])
        ).all()

        currents = torch.zeros(6)
        nexts = torch.zeros(6)
        stage_times = torch.zeros(6)
        n = 1000
        for i in range(n):
            current_stage = torch.tensor([0, 1, 2, 3, 4, 5])
            next_stage = torch.tensor([0, 1, 3, 4, 5, 5])
            time_to_next_stage = torch.tensor([1.1, 2.5, 0.7, 0.9, 0.1, 0.5])
            current, next, stage_time = sp.sample_next_stage(
                ages=ages,
                symptoms_susceptibility=symptoms_susceptibility,
                infection_ids=infection_ids,
                current_stage=current_stage,
                next_stage=next_stage,
                time_to_next_stage=time_to_next_stage,
                time=time,
            )
            currents += current
            nexts += next
            stage_times += stage_time

        # Check new current stages
        currents = currents / n
        currents = currents.numpy()
        assert currents[0] == 0
        assert currents[1] == 1
        assert currents[2] == 3
        assert currents[3] == 4
        assert currents[4] == 5
        assert currents[5] == 5

        # Check new next stages
        nexts = nexts / n
        nexts = nexts.numpy()
        assert nexts[0] == 0
        assert nexts[1] == 1
        assert np.isclose(nexts[2], 4 * 0.2, rtol=0.1)
        assert np.isclose(nexts[3], 5 * 0.1, rtol=0.1)
        assert nexts[4] == 5
        assert nexts[5] == 5

        stage_times = stage_times / n
        stage_times = stage_times.numpy()
        assert np.isclose(stage_times[0], 1.1, rtol=1e-1)
        assert np.isclose(stage_times[1], 2.5, rtol=1e-1)
        expected = 0.7 + 0.2 * 10.2 + 0.8 * np.exp(1.3 + 0.5**2 / 2)
        assert np.isclose(stage_times[2], expected, rtol=1e-1)
        expected = (
            0.9 + 0.1 * np.exp(1.7 + 0.5**2 / 2) + 0.9 * np.exp(1.4 + 0.8**2 / 2)
        )
        assert np.isclose(stage_times[3], expected, rtol=1e-1)
        assert np.isclose(stage_times[4], 0.1, rtol=1e-1)
        assert stage_times[5] == 0.5


class TestSymptomsUpdater:
    @fixture(name="sp")
    def make_sp(self):
        return SymptomsSampler.from_file()

    @fixture(name="su")
    def make_su(self, sp):
        return SymptomsUpdater(sp)

    def test__update_symptoms(self, su, data, timer):
        n_agents = len(data["agent"].id)
        data["agent"]["symptoms"]["current_stage"] = 2 * torch.ones(
            n_agents, dtype=torch.long
        )
        data["agent"]["symptoms"]["next_stage"] = 3 * torch.ones(
            n_agents, dtype=torch.long
        )  # all infectious
        data["agent"]["symptoms"]["time_to_next_stage"] = torch.zeros(n_agents)
        symptoms = su(
            data=data, timer=timer, new_infected=torch.zeros(n_agents, dtype=torch.bool)
        )
        assert (
            symptoms["current_stage"] == 3 * torch.ones(n_agents, dtype=torch.long)
        ).all()
        will_recover = len(symptoms["next_stage"][symptoms["next_stage"] == 0])
        assert will_recover < n_agents / 2
        will_symptom = len(symptoms["next_stage"][symptoms["next_stage"] == 4])
        assert will_symptom > n_agents / 2
        assert will_recover + will_symptom == n_agents

    def test__dead_stay_dead(self, su, data, timer):
        n_agents = len(data["agent"].id)
        data["agent"]["symptoms"]["current_stage"] = 6 * torch.ones(
            n_agents, dtype=torch.long
        )
        data["agent"]["symptoms"]["next_stage"] = 7 * torch.ones(
            n_agents, dtype=torch.long
        )  # all infectious
        data["agent"]["symptoms"]["time_to_next_stage"] = torch.zeros(n_agents)
        for i in range(100):
            symptoms = su(
                data=data,
                timer=timer,
                new_infected=torch.zeros(n_agents, dtype=torch.bool),
            )
            assert (
                symptoms["current_stage"] == 7 * torch.ones(n_agents, dtype=torch.long)
            ).all()
            next(timer)

    def test__symptoms_differentiable(self, su, data, timer):
        # increase mortality for the test
        su.symptoms_sampler.stage_transition_probabilities[2:, :] = torch.ones(
            su.symptoms_sampler.stage_transition_probabilities[2:, :].shape
        )
        beta = torch.nn.Parameter(torch.tensor(10.0))
        probs = 1 - torch.exp(-beta) * torch.ones(data["agent"].id.shape)
        new_infected = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=torch.tensor(0.1),
            probs=probs,
        ).rsample()
        symptoms = su(data=data, timer=timer, new_infected=new_infected)
        new_infected_2 = torch.zeros(new_infected.shape)
        for _ in range(100):
            next(timer)
            symptoms = su(data=data, timer=timer, new_infected=new_infected_2)
        deaths = symptoms["current_stage"][symptoms["current_stage"] == 7]
        assert len(deaths) > 0
        total_deaths = deaths.sum()
        total_deaths.backward()
        assert deaths.requires_grad
        assert symptoms["current_stage"].requires_grad
        assert symptoms["time_to_next_stage"].requires_grad
        assert symptoms["next_stage"].requires_grad
