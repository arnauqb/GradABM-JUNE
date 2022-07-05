import torch
import yaml

from torch_june.paths import default_config_path
from torch_june.utils import parse_distribution


class TransmissionSampler:
    def __init__(self, max_infectiousness, shape, rate, shift):
        self.max_infectiousness = max_infectiousness
        self.shape = shape
        self.rate = rate
        self.shift = shift

    def __call__(self, n):
        maxi = self.max_infectiousness.rsample((n,))
        shape = self.shape.rsample((n,))
        rate = self.rate.rsample((n,))
        shift = self.shift.rsample((n,))
        return torch.vstack((maxi, shape, rate, shift))

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        ret = {}
        device = params["system"]["device"]
        tparams = params["transmission"]
        for variant in tparams:
            ret[variant] = {}
            for key in tparams[variant]:
                ret[variant][key] = parse_distribution(tparams[key], device=device)
        return cls(**ret)


class TransmissionUpdater(torch.nn.Module):
    def forward(self, data, timer):
        time_from_infection = timer.now - data["agent"].infection_time
        ret = None
        for infection_variant in data["infection_parameters"]["variants"]:
            inf_params = data["agent"]["infection_parameters"]
            shape = inf_params.get(infection_variant, inf_params["base"])["shape"]
            shift = inf_params.get(infection_variant, inf_params["base"])["shift"]
            rate = inf_params.get(infection_variant, inf_params["base"])["rate"]
            max_infectiousness = inf_params.get(infection_variant, inf_params["base"])[
                "max_infectiousness"
            ]
            sign = (torch.sign(time_from_infection - shift + 1e-10) + 1) / 2
            aux = torch.exp(-torch.lgamma(shape)) * torch.pow(
                (time_from_infection - shift) * rate, shape - 1.0
            )
            aux2 = torch.exp((shift - time_from_infection) * rate) * rate
            variant_infectivity = (
                max_infectiousness * sign * aux * aux2 * data["agent"].is_infected
            )
            if ret is None:
                ret = variant_infectivity
            else:
                ret = torch.vstack((ret, variant_infectivity))
        return ret
