from pytest import fixture
import numpy as np
import torch

from torch_june.symptoms import SymptomsSampler


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
            "transition_probabilities": {
                "asymptomatic": {"0-50": 1.0, "50-100": 0.5},
                "symptomatic": {"0-100": 0.2},
                "critical": {"0-100": 0.1},
            },
            "symptom_transition_times": {
                "asymptomatic": {"distribution": "LogNormal", "loc": 1.1, "scale": 0.2},
                "symptomatic": {"distribution": "Normal", "loc": 10.2, "scale": 3.0},
                "critical": {"distribution": "LogNormal", "loc": 1.7, "scale": 0.5},
            },
            "recovery_times": {
                "asymptomatic": {"distribution": "LogNormal", "loc": 1.2, "scale": 0.3},
                "symptomatic": {"distribution": "LogNormal", "loc": 1.3, "scale": 0.5},
                "critical": {"distribution": "LogNormal", "loc": 1.4, "scale": 0.8},
            },
        }
        return input

    @fixture(name="sp")
    def make_sp(self, input):
        return SymptomsSampler.from_dict(input)

    def test__read_input(self, sp):
        assert sp.stages == [
            "recovered",
            "susceptible",
            "asymptomatic",
            "symptomatic",
            "critical",
            "dead",
        ]
        assert (sp.transition_probabilities[0] == torch.zeros(100)).all()
        assert (sp.transition_probabilities[1] == torch.zeros(100)).all()
        assert (sp.transition_probabilities[2][:50] == 1.0 * torch.ones(50)).all()
        assert (sp.transition_probabilities[2][50:] == 0.5 * torch.ones(50)).all()
        assert (sp.transition_probabilities[3] == 0.2 * torch.ones(100)).all()
        assert (sp.transition_probabilities[4] == 0.1 * torch.ones(100)).all()

        assert sp.symptom_transition_times[2].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.symptom_transition_times[2].loc.item(), 1.1)
        assert np.isclose(sp.symptom_transition_times[2].scale.item(), 0.2)
        assert np.isclose(
            sp.symptom_transition_times[2].mean.item(), np.exp(1.1 + 0.2**2 / 2)
        )
        assert sp.symptom_transition_times[3].__class__.__name__ == "Normal"
        assert np.isclose(sp.symptom_transition_times[3].loc.item(), 10.2)
        assert np.isclose(sp.symptom_transition_times[3].scale.item(), 3.0)
        assert np.isclose(sp.symptom_transition_times[3].mean.item(), 10.2)
        assert sp.symptom_transition_times[4].__class__.__name__ == "LogNormal"
        assert np.isclose(sp.symptom_transition_times[4].loc.item(), 1.7)
        assert np.isclose(sp.symptom_transition_times[4].scale.item(), 0.5)
        assert np.isclose(
            sp.symptom_transition_times[4].mean.item(), np.exp(1.7 + 0.5**2 / 2)
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
        current_stages = torch.tensor([0, 1, 2, 3, 4, 5])
        # next_stages = torch.tensor([0, 1, 3, 4, 5, 5])
        time_to_next_stages = torch.tensor([1.1, 2.5, 0.7, 1.8, 0.1, 0.5])

        time = 1.0
        need_to_trans = sp._get_need_to_transition(
            current_stages, time_to_next_stages, time
        )
        assert (need_to_trans == torch.tensor([0, 0, 1, 0, 1, 0]).to(torch.bool)).all()

        probability_next_symptomatic_stage = sp._get_prob_next_symptoms_stage(
            ages, current_stages
        )
        assert (
            probability_next_symptomatic_stage
            == torch.tensor([0.0, 0.0, 1.0, 0.2, 0.1, 0])
        ).all()

        transitions = torch.zeros(6)
        transition_times = torch.zeros(6)
        n = 5000
        for i in range(n):
            trans, trans_t = sp.sample_next_stages(
                ages, current_stages, time_to_next_stages, time
            )
            transitions += trans
            transition_times += trans_t
        transitions = transitions / n
        transitions = transitions.numpy()
        assert transitions[0] == 0
        assert transitions[1] == 1
        expected = 3
        assert np.isclose(transitions[2], expected)
        assert transitions[3] == 3
        expected = 0.1 * 5 + 0.9 * 0
        assert np.isclose(transitions[4], expected, rtol=1e-1)
        assert transitions[5] == 5

        transition_times = transition_times / n
        transition_times = transition_times.numpy()
        assert np.isclose(transition_times[0], 1.1, rtol=1e-2)
        assert np.isclose(transition_times[1], 2.5, rtol=1e-2)
        expected = 10.2 + 0.7
        assert np.isclose(transition_times[2], expected, rtol=1e-1)
        assert np.isclose(transition_times[3], 1.8, rtol=1e-2)
        expected = 0.1 * np.exp(1.7 + 0.5**2/2) + 0.9 * np.exp(1.4 + 0.8**2/2)
        assert np.isclose(transition_times[4], expected, rtol=1e-1)
        assert transition_times[5] == 0.5
