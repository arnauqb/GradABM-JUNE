import torch
import yaml

from grad_june.paths import default_config_path
from grad_june.utils import parse_distribution


class TransmissionSampler:
    def __init__(self, infection_names, max_infectiousness, shape, rate, shift, device):
        """
        Samples transmission parameters per infection type.

        Parameters
        ----------
        infection_names:
            list with the names of the infections, eg, ["base", "delta", "omicron"]
        max_infectiousness:
            maximum value of infectiousness
        shape:
            shape parameter of the Gamma distribution. This should have the same dimension as the number of infections.
        rate:
            rate parameter of the Gamma distribution. This should have the same dimension as the number of infections.
        shift:
            shift parameter of the Gamma distribution. This should have the same dimension as the number of infections.
        """
        self.n_infections = len(infection_names)
        self.infection_names = infection_names
        self.max_infectiousness = max_infectiousness
        self.shape = shape
        self.rate = rate
        self.shift = shift
        self.device = device

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        param_data = {}
        device = params["system"]["device"]
        tparams = params["transmission"]
        for variant in tparams:
            param_data[variant] = {}
            for key in tparams[variant]:
                param_data[variant][key] = parse_distribution(
                    tparams[variant][key], device=device
                )
        # autofill missing values with base default
        for variant in tparams:
            if variant == "base":
                continue
            for key in tparams["base"]:
                if key not in param_data[variant]:
                    param_data[variant][key] = param_data["base"][key]
        # build tensors
        parameter_names = [key for key in param_data["base"]]
        infection_names = list(tparams.keys())
        n_infections = len(infection_names)
        ret = {}
        for pname in parameter_names:
            ret[pname] = [param_data[variant][pname] for variant in tparams]
        return cls(**ret, infection_names=infection_names, device=device)

    def __call__(self, n_agents):
        ret = {
            "n_infections": self.n_infections,
            "infection_ids": torch.arange(0, self.n_infections),
            "infection_names": self.infection_names,
        }
        for pname in ["max_infectiousness", "shape", "rate", "shift"]:
            ret[pname] = torch.zeros((self.n_infections, n_agents), device=self.device)
            for i in range(self.n_infections):
                ret[pname][i] = getattr(self, pname)[i].rsample((n_agents,))
        return ret


class TransmissionUpdater(torch.nn.Module):
    """
    Updates the infectivity value for each agent / variant over time.
    """

    def forward(self, data, timer):
        time_from_infection = timer.now - data["agent"].infection_time
        inf_params = data["agent"]["infection_parameters"]
        agent_infection_ids = data["agent"].infection_id.reshape(1, -1)
        shape = torch.gather(inf_params["shape"], 0, agent_infection_ids).flatten()
        shift = torch.gather(inf_params["shift"], 0, agent_infection_ids).flatten()
        rate = torch.gather(inf_params["rate"], 0, agent_infection_ids).flatten()
        max_infectiousness = torch.gather(
            inf_params["max_infectiousness"], 0, agent_infection_ids
        ).flatten()
        sign = (torch.sign(time_from_infection - shift + 1e-10) + 1) / 2
        # TODO: Currently lgamma is not supported on M1 macs.
        if shape.device.type == "mps":
            gg = torch.lgamma(shape.cpu()).to(shape.device)
        else:
            gg = torch.lgamma(shape)
        aux = torch.exp(-gg) * torch.pow(
            (time_from_infection - shift) * rate, shape - 1.0
        )
        aux2 = torch.exp((shift - time_from_infection) * rate) * rate
        ret = max_infectiousness * sign * aux * aux2 * data["agent"].is_infected
        return ret
