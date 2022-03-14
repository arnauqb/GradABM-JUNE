import numpy as np
import torch
from pytest import fixture

from torch_june.infections import Infections


class TestInfections:
    def test__sampler(self, sampler):
        assert np.isclose(sampler.max_infectiousness.mean, 1.1331, rtol=1e-3)
        assert sampler.shape.mean == 1.56
        assert sampler.rate.mean == 0.53
        assert sampler.shift.mean == -2.12
        assert sampler(10).shape == (4, 10)

    def test__generate_infections(self, sampler):
        parameters = sampler(10)
        infections = Infections(parameters)
        time = 0
        assert (infections.infection_times == -1.0 * torch.ones(10)).all()
        assert (infections.get_transmissions(time) == torch.zeros(10)).all()
        is_infected = torch.hstack((torch.ones(1), torch.zeros(9)))
        infection_time = 2.0
        infections.update(is_infected, infection_time)
        assert infections.is_infected[0] == 1
        assert infections.infection_times[0] == 2.0
        for i in range(1,10):
            assert infections.is_infected[i] == 0
            assert infections.infection_times[i] == -1.0
        is_infected = torch.hstack((torch.zeros(9), torch.ones(1)))
        infection_time = 3.0
        infections.update(is_infected, infection_time)
        assert infections.is_infected[0] == 1
        assert infections.infection_times[0] == 2.0
        assert infections.is_infected[9] == 1
        assert infections.infection_times[9] == 3.0
        for i in range(1,9):
            assert infections.is_infected[i] == 0
            assert infections.infection_times[i] == -1.0

        time = 5
        trans = infections.get_transmissions(time)
        for i in range(10):
            if i in [0, 9]:
                assert trans[i] > 0.0
            else:
                assert trans[i] == 0.0
