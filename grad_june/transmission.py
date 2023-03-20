import torch
import yaml

from grad_june.paths import default_config_path
from grad_june.utils import parse_distribution


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
        tparams = params["transmission"]
        device = params["system"]["device"]
        for key in tparams:
            ret[key] = parse_distribution(tparams[key], device=device)
        return cls(**ret)


class TransmissionUpdater(torch.nn.Module):
    def forward(self, data, timer):
        shape = data["agent"]["infection_parameters"]["shape"]
        shift = data["agent"]["infection_parameters"]["shift"]
        rate = data["agent"]["infection_parameters"]["rate"]
        max_infectiousness = data["agent"]["infection_parameters"]["max_infectiousness"]
        time_from_infection = timer.now - data["agent"].infection_time
        sign = (torch.sign(time_from_infection - shift + 1e-10) + 1) / 2
        aux = torch.exp(-torch.lgamma(shape)) * torch.pow(
            (time_from_infection - shift) * rate, shape - 1.0
        )
        aux2 = torch.exp((shift - time_from_infection) * rate) * rate
        ret = max_infectiousness * sign * aux * aux2 * data["agent"].is_infected
        return ret
