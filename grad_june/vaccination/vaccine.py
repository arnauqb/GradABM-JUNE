import yaml
import torch
from copy import deepcopy

from grad_june.paths import default_config_path
from grad_june.utils import parse_age_probabilities


class Vaccines:
    def __init__(
        self, names, sterilization_efficacies, symptomatic_efficacies, coverages
    ):
        self.ids = torch.arange(len(names))
        self.names = names
        self.sterilization_efficacies = sterilization_efficacies
        self.symptomatic_efficacies = symptomatic_efficacies
        self.coverages = coverages

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params["vaccines"])

    @classmethod
    def from_parameters(cls, params):
        names = list(params.keys())
        (
            sterilization_efficacies,
            symptomatic_efficacies,
            coverages,
        ) = cls._parse_parameters(params)
        return cls(
            names=names,
            sterilization_efficacies=sterilization_efficacies,
            symptomatic_efficacies=symptomatic_efficacies,
            coverages=coverages,
        )

    @classmethod
    def _parse_parameters(cls, params):
        params = deepcopy(params)
        sterilization_efficacies = []
        symptomatic_efficacies = []
        coverages = []
        for name in params:
            # coverage
            vax_coverage = params[name].pop("coverage")
            age_coverages = parse_age_probabilities(vax_coverage)
            coverages.append(age_coverages)
            # efficacies
            sterilization_efficacies_ = []
            symptomatic_efficacies_ = []
            for variant in params[name]:
                sterilization_efficacies_.append(
                    params[name][variant].get(
                        "sterilization_efficacy",
                        params[name]["base"]["sterilization_efficacy"],
                    )
                )
                symptomatic_efficacies_.append(
                    params[name][variant].get(
                        "symptomatic_efficacy",
                        params[name]["base"]["symptomatic_efficacy"],
                    )
                )
            sterilization_efficacies.append(sterilization_efficacies_)
            symptomatic_efficacies.append(symptomatic_efficacies_)
        # no vax option for coverages.
        coverages = torch.tensor(coverages)
        no_vax = 1.0 - coverages.sum(0)
        coverages = torch.vstack((no_vax, coverages))
        sterilization_efficacies = torch.tensor(
            sterilization_efficacies, dtype=torch.float64
        )
        sterilization_efficacies = torch.vstack(
            (torch.zeros(sterilization_efficacies.shape[1]), sterilization_efficacies)
        )
        symptomatic_efficacies = torch.tensor(
            symptomatic_efficacies, dtype=torch.float64
        )
        symptomatic_efficacies = torch.vstack(
            (torch.zeros(symptomatic_efficacies.shape[1]), symptomatic_efficacies)
        )
        return sterilization_efficacies, symptomatic_efficacies, coverages

    def sample_efficacies(self, ages):
        coverage_probs = self.coverages[:, ages]
        sample_vax = torch.nn.functional.gumbel_softmax(
            torch.log(coverage_probs), dim=0, tau=0.1, hard=True
        )
        ster_efficacies = torch.matmul(
            self.sterilization_efficacies.transpose(0, 1), sample_vax
        )
        symp_efficacies = torch.matmul(
            self.symptomatic_efficacies.transpose(0, 1), sample_vax
        )
        return ster_efficacies, symp_efficacies
