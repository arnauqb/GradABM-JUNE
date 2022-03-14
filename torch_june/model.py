from torch.nn.parameter import Parameter
import torch

from torch_june import InfectionPassing


class TorchJune(torch.nn.Module):
    def __init__(self, betas, data, device="cpu"):
        super().__init__()
        self.data = data.to(device)
        self._betas_to_idcs = {name: i for i, name in enumerate(betas.keys())}
        self.device = device
        self.beta_parameters = Parameter(torch.tensor(list(betas.values())))
        self.inf_network = InfectionPassing(device=device)

    def _get_edge_types_from_timer(self, timer):
        ret = []
        for activity in timer.activities:
            ret.append("attends_" + activity)
        return ret

    def forward(self, timer, transmissions, susceptibilities):
        ret = None
        betas = {
            beta_n: self.beta_parameters[self._betas_to_idcs[beta_n]]
            for beta_n in self._betas_to_idcs.keys()
        }
        while timer.date < timer.final_date:
            infection_probs = self.inf_network(
                data=self.data,
                edge_types=self._get_edge_types_from_timer(timer),
                betas=betas,
                delta_time = timer.duration,
                transmissions=transmissions,
                susceptibilities=susceptibilities,
            )
            new_infected = self.inf_network.sample_infected(infection_probs)
            if ret is None:
                ret = new_infected
            else:
                ret = torch.vstack((ret, new_infected))
            transmissions = transmissions + 0.2 * new_infected
            susceptibilities = susceptibilities - new_infected
            next(timer)

        return ret