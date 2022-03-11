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

    def forward(self, n_timesteps, transmissions, susceptibilities):
        ret = None
        betas = {
            beta_n: self.beta_parameters[self._betas_to_idcs[beta_n]]
            for beta_n in self._betas_to_idcs.keys()
        }
        for _ in range(n_timesteps):
            infection_probs = self.inf_network(
                data=self.data,
                edge_types=("attends_household", "attends_company", "attends_school"),
                betas=betas,
                transmissions=transmissions,
                susceptibilities = susceptibilities
            )
            new_infected = self.inf_network.sample_infected(infection_probs)
            if ret is None:
                ret = new_infected
            else:
                ret = torch.vstack((ret, new_infected))
            transmissions = transmissions + 0.2 * new_infected
            susceptibilities = susceptibilities - new_infected

        return ret
