import numpy as np

from grad_june.infection import (
    infect_people_at_indices,
    infect_fraction_of_people,
)
from grad_june.timer import Timer
from grad_june.symptoms import SymptomsUpdater


class TestInfectionSeed:
    def test__simple_seed(self, data):
        data = infect_people_at_indices(data, [0, 2, 5])
        for i in range(len(data["agent"].id)):
            if i in [0, 2, 5]:
                assert data["agent"]["susceptibility"][0, i] == 0.0
                assert data["agent"]["susceptibility"][1, i] == 0.0
                assert data["agent"]["susceptibility"][2, i] == 0.0
                assert data["agent"]["is_infected"][i] == 1
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 2.0
            else:
                assert data["agent"]["susceptibility"][0, i] == 1.0
                assert data["agent"]["susceptibility"][1, i] == 1.0
                assert data["agent"]["susceptibility"][2, i] == 1.0
                assert data["agent"]["is_infected"][i] == 0
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 1.0

    def test__differentiable_seed(self, data):
        su = SymptomsUpdater.from_file()
        timer = Timer.from_file()
        infect_fraction_of_people(
            data=data,
            timer=timer,
            symptoms_updater=su,
            fraction=0.3,
            device="cpu",
            infection_type=0,
        )
        assert np.isclose(
            data["agent"].is_infected.sum(), 0.3 * data["agent"].id.shape[0], rtol=3e-1
        )
        for i in range(len(data["agent"].id)):
            if data["agent"].is_infected[i]:
                # Symptoms updater should be called afterwards to set the right symptoms
                assert data["agent"]["susceptibility"][0, i] == 0.0
                assert data["agent"]["susceptibility"][1, i] == 0.0
                assert data["agent"]["susceptibility"][2, i] == 0.0
                assert data["agent"]["is_infected"][i] == 1
                assert data["agent"]["infection_time"][i] == 0.0
                assert (
                    data["agent"]["symptoms"]["next_stage"][i] == 1.0
                )  # this is changed later with symptoms updater.
            else:
                assert data["agent"]["susceptibility"][0, i] == 1.0
                assert data["agent"]["susceptibility"][1, i] == 1.0
                assert data["agent"]["susceptibility"][2, i] == 1.0
                assert data["agent"]["is_infected"][i] == 0
                assert data["agent"]["infection_time"][i] == 0.0
                assert data["agent"]["symptoms"]["next_stage"][i] == 1.0
