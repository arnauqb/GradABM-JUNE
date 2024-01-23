import numpy as np
import torch
from pytest import fixture

from grad_june.transmission import TransmissionUpdater


class TestInfections:
    def test__sampler(self, sampler):
        assert np.isclose(sampler.max_infectiousness.mean, 1.1331, rtol=1e-3)
        assert sampler.shape.mean == 1.56
        assert sampler.rate.mean == 0.53
        assert sampler.shift.mean == 2.12
        assert sampler(10).shape == (4, 10)

    def test__generate_infections(self, data, timer):
        trans_updater = TransmissionUpdater()
        transmissions = trans_updater(data=data, time=timer.now)
        assert (transmissions == torch.zeros(100)).all()
        is_infected = np.zeros(100)
        infection_time = -1.0 * np.ones(100)
        is_infected[0] = 1
        is_infected[-1] = 1
        infection_time[0] = 2.0
        infection_time[-1] = 3.0
        data["agent"].is_infected = torch.tensor(is_infected)
        data["agent"].infection_time = torch.tensor(infection_time)
        while timer.now != 5:
            next(timer)
        transmissions = trans_updater(data=data, time=timer.now)
        assert (transmissions[1:99] == torch.zeros(98)).all()
        assert transmissions[0] > 0.0
        assert transmissions[99] > 0.0
