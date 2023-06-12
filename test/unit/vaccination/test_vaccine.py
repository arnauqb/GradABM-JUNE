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

    @pytest.fixture(name="vaccines")
    def make_vax(self, config):
        vaccines = Vaccines.from_parameters(config, device="cpu")
        return vaccines

    def test__init_vaccines(self, vaccines):
        assert vaccines.names == ["pfizer", "astrazeneca"]
        assert vaccines.ids.equal(torch.tensor([0, 1]))
        assert torch.allclose(
            vaccines.sterilization_efficacies,
            torch.tensor([[0, 0], [0.7, 0.3], [0.75, 0.1]], dtype=torch.float),
        )
        assert torch.allclose(
            vaccines.symptomatic_efficacies,
            torch.tensor([[0, 0], [0.9, 0.8], [0.8, 0.8]], dtype=torch.float),
        )
        assert torch.allclose(
            vaccines.coverages[0, 0:50], 0.15 * torch.ones(50, dtype=torch.float)
        )
        assert torch.allclose(
            vaccines.coverages[0, 50:], 0.1 * torch.ones(50, dtype=torch.float)
        )
        assert torch.allclose(
            vaccines.coverages[1, 0:50], 0.7 * torch.ones(50, dtype=torch.float)
        )
        assert torch.allclose(
            vaccines.coverages[1, 50:], 0.3 * torch.ones(50, dtype=torch.float)
        )
        assert torch.allclose(
            vaccines.coverages[2, 0:50], 0.15 * torch.ones(50, dtype=torch.float)
        )
        assert torch.allclose(
            vaccines.coverages[2, 50:], 0.6 * torch.ones(50, dtype=torch.float)
        )

    def test__vaccine_efficacies(self, vaccines):
        ages = torch.tensor([10, 20, 80])
        ret_ster = torch.zeros((2, 3))
        ret_symp = torch.zeros((2, 3))
        n = 1000
        for i in range(n):
            ster, symp = vaccines.sample_efficacies(ages=ages)
            ret_ster += ster
            ret_symp += symp
        ret_ster = ret_ster / n
        ret_symp = ret_symp / n
        rtol = 0.05
        assert torch.allclose(
            ret_ster[0, :],
            torch.tensor(
                [
                    0.7 * 0.7 + 0.15 * 0.75,
                    0.7 * 0.7 + 0.15 * 0.75,
                    0.3 * 0.7 + 0.6 * 0.75,
                ]
            ),
            rtol=rtol,
        )
        assert torch.allclose(
            ret_ster[1, :],
            torch.tensor(
                [
                    0.7 * 0.3 + 0.15 * 0.10,
                    0.7 * 0.3 + 0.15 * 0.10,
                    0.3 * 0.3 + 0.6 * 0.10,
                ]
            ),
            rtol=rtol,
        )

        assert torch.allclose(
            ret_symp[0, :],
            torch.tensor(
                [
                    0.7 * 0.9 + 0.15 * 0.80,
                    0.7 * 0.9 + 0.15 * 0.80,
                    0.3 * 0.9 + 0.6 * 0.80,
                ]
            ),
            rtol=rtol,
        )
        assert torch.allclose(
            ret_symp[1, :],
            torch.tensor(
                [
                    0.7 * 0.8 + 0.15 * 0.80,
                    0.7 * 0.8 + 0.15 * 0.80,
                    0.3 * 0.8 + 0.6 * 0.80,
                ]
            ),
            rtol=rtol,
        )

    def test_vaccination(self, vaccines):
        ages = torch.tensor([10, 20, 80])
        susceptibilities_avg = torch.zeros(2, 3)
        symp_susceptibilities_avg = torch.zeros(2, 3)
        n = 1000
        for i in range(n):
            susceptibilities = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
            symptom_susceptibilities = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
            susceptibilities, symptom_susceptibilities = vaccines.vaccinate(
                ages=ages,
                susceptibilities=susceptibilities,
                symptom_susceptibilities=symptom_susceptibilities,
            )
            susceptibilities_avg += susceptibilities
            symp_susceptibilities_avg += symptom_susceptibilities
        susceptibilities_avg = susceptibilities_avg / n
        symp_susceptibilities_avg = symp_susceptibilities_avg / n
        rtol = 0.05
        assert torch.allclose(
            susceptibilities_avg[0, :],
            torch.tensor(
                [
                    1.0 - (0.7 * 0.7 + 0.15 * 0.75),
                    1.0 - (0.7 * 0.7 + 0.15 * 0.75),
                    1.0 - (0.3 * 0.7 + 0.6 * 0.75),
                ]
            ),
            rtol=rtol,
        )
        assert torch.allclose(
            susceptibilities_avg[1, :],
            torch.tensor(
                [
                    1.0 - (0.7 * 0.3 + 0.15 * 0.10),
                    1.0 - (0.7 * 0.3 + 0.15 * 0.10),
                    1.0 - (0.3 * 0.3 + 0.6 * 0.10),
                ]
            ),
            rtol=rtol,
        )

        assert torch.allclose(
            symp_susceptibilities_avg[0, :],
            torch.tensor(
                [
                    1.0 - (0.7 * 0.9 + 0.15 * 0.80),
                    0.0,
                    1.0 - (0.3 * 0.9 + 0.6 * 0.80),
                ]
            ),
            rtol=rtol,
        )
        assert torch.allclose(
            symp_susceptibilities_avg[1, :],
            torch.tensor(
                [
                    1.0 - (0.7 * 0.8 + 0.15 * 0.80),
                    0.0,
                    1.0 - (0.3 * 0.8 + 0.6 * 0.80),
                ]
            ),
            rtol=rtol,
        )
