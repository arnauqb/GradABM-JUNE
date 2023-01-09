import numpy as np
import torch

from grad_june.infection_networks.is_infected_sampler import IsInfectedSampler

class TestIsInfectedSampler:
    def test__sample_infected(self):
        sampler = IsInfectedSampler()
        probs = torch.tensor([0.3])
        n = 2000
        ret = 0
        for _ in range(n):
            ret += sampler(probs, 0)
        ret = ret / n
        assert np.isclose(ret, 0.7, rtol=1e-1)

        probs = torch.tensor([0.2, 0.5, 0.7, 0.3])
        n = 2000
        ret = torch.zeros(4)
        for _ in range(n):
            ret += sampler(probs, 0)

        ret = ret / n
        assert np.allclose(ret, 1.0 - probs, rtol=1e-1)

