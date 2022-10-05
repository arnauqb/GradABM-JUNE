import numpy as np

from torch_june.infection_seed import (
    infect_people_at_indices,
    infect_fraction_of_people,
)
from torch_june.timer import Timer
from torch_june.symptoms import SymptomsUpdater


class TestInfectionSeed:
    def test__simple_seed(self, data):
        data = infect_people_at_indices(data, [0, 2, 5])
        for i in range(len(data["agent"].id)):
            if i in [0, 2, 5]:
                assert data["agent"]["susceptibility"][i] == 0.0
                assert data["agent"]["is_infected"][i] == 1
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 2.0
            else:
                assert data["agent"]["susceptibility"][i] == 1.0
                assert data["agent"]["is_infected"][i] == 0
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 1.0

    def test__differentiable_seed(self, data):
        su = SymptomsUpdater.from_file()
        timer = Timer.from_file()
        infect_fraction_of_people(
            data=data, timer=timer, symptoms_updater=su, fraction=0.2, device="cpu"
        )
        assert np.isclose(
            data["agent"].is_infected.sum(), 0.2 * data["agent"].id.shape[0], rtol=1e-1
        )
        for i in range(len(data["agent"].id)):
            if data["agent"].is_infected[i]:
                # Symptoms updater should be called afterwards to set the right symptoms
                assert data["agent"]["susceptibility"][i] == 0.0
                assert data["agent"]["is_infected"][i] == 1
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 1.0 # this is changed later with symptoms updater.
            else:
                assert data["agent"]["susceptibility"][i] == 1.0
                assert data["agent"]["is_infected"][i] == 0
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 1.0
