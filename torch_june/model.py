from torch.nn.parameter import Parameter
import time
import torch

from torch_june import InfectionPassing


class TorchJune(torch.nn.Module):
    def __init__(self, betas, data, infections, device="cpu"):
        super().__init__()
        self.data = data.to(device)
        self._betas_to_idcs = {name: i for i, name in enumerate(betas.keys())}
        self.device = device
        self.beta_parameters = Parameter(torch.tensor(list(betas.values())))
        self.inf_network = InfectionPassing(device=device)
        self.infections = infections

    def _get_edge_types_from_timer(self, timer):
        ret = []
        for activity in timer.activities:
            ret.append("attends_" + activity)
        return ret

    def forward(self, timer, susceptibilities):
        ret = None
        betas = {
            beta_n: self.beta_parameters[self._betas_to_idcs[beta_n]]
            for beta_n in self._betas_to_idcs.keys()
        }
        while timer.date < timer.final_date:
            transmissions = self.infections.get_transmissions(time=timer.now)
            infection_probs = self.inf_network(
                data=self.data,
                edge_types=self._get_edge_types_from_timer(timer),
                betas=betas,
                delta_time=timer.duration,
                transmissions=transmissions,
                susceptibilities=susceptibilities,
            )
            new_infected = self.inf_network.sample_infected(infection_probs)
            self.infections.update(new_infected=new_infected, infection_time=timer.now)
            if ret is None:
                ret = new_infected
            else:
                ret = torch.vstack((ret, new_infected))
            next(timer)
            susceptibilities = susceptibilities - new_infected

        return ret
