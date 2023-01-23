import pytest
import torch

from grad_june.symptoms import SymptomsSampler, SymptomsUpdater
from grad_june.vaccination import Vaccines


class TestVaccine:
    @pytest.fixture(name="config")
    def make_config(self):
        return {
            "pfizer": {
                "base": {"symptomatic_efficacy": 0.9, "sterilization_efficacy": 0.7},
                "delta": {"symptomatic_efficacy": 0.8, "sterilization_efficacy": 0.3},
                "coverage": {"0-50": 0.7, "50-100": 0.3},
            },
            "astrazeneca": {
                "base": {"symptomatic_efficacy": 0.8, "sterilization_efficacy": 0.75},
                "delta": {"sterilization_efficacy": 0.1},
                "coverage": {"0-50": 0.15, "50-100": 0.6},
            },
        }

    def test__init_vaccines(self, config):
        vaccines = Vaccines.from_parameters(config)
        assert vaccines.names == ["pfizer", "astrazeneca"]
        assert vaccines.ids.equal(torch.tensor([0, 1]))
        assert vaccines.sterilization_efficacies.equal(
            torch.tensor([[0.7, 0.3], [0.75, 0.1]])
        )
        assert vaccines.symptomatic_efficacies.equal(
            torch.tensor([[0.9, 0.8], [0.8, 0.8]])
        )
        assert torch.allclose(
            vaccines.coverages[0, 0:50], 0.7 * torch.ones(50, dtype=torch.float64)
        )
        assert torch.allclose(
            vaccines.coverages[0, 50:], 0.3 * torch.ones(50, dtype=torch.float64)
        )
        assert torch.allclose(
            vaccines.coverages[1, 0:50], 0.15 * torch.ones(50, dtype=torch.float64)
        )
        assert torch.allclose(
            vaccines.coverages[1, 50:], 0.6 * torch.ones(50, dtype=torch.float64)
        )

    def test__symptoms_susceptibility(self, data):
        pass
