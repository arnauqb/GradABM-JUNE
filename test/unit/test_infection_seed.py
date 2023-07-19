import numpy as np
import torch

from grad_june.infection_seed import (
    infect_people_at_indices,
    InfectionSeedByDistrict,
    InfectionSeedByFraction
)
from grad_june.timer import Timer
from grad_june.symptoms import SymptomsUpdater
from grad_june.demographics import get_people_per_area


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
        seed = InfectionSeedByFraction(0.2, device="cpu")
        seed(data, 0)
        assert torch.isclose(
            data["agent"].is_infected.sum(), torch.tensor(0.2 * data["agent"].id.shape[0]), rtol=2e-1
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

    def test__seeding_by_district(self, data):
        cases_per_district = {0: [3, 5, 2], 1: [7, 2, 4], 2: [2, 3, 4]}
        seed = InfectionSeedByDistrict(cases_per_district, device="cpu")
        seed(data, 0)
        assert data["agent"].is_infected.sum() == 12
        people_per_district = get_people_per_area(data["agent"].id, data["agent"].district_id)
        for i, district in enumerate(people_per_district):
            infected = 0
            for person_id in people_per_district[district]:
                if data["agent"].is_infected[person_id]:
                    infected += 1
            assert infected == cases_per_district[i][0]
        seed(data, 1)
        assert data["agent"].is_infected.sum() == 12 + 10
        for i, district in enumerate(people_per_district):
            infected = 0
            for person_id in people_per_district[district]:
                if data["agent"].is_infected[person_id]:
                    infected += 1
            assert infected == sum(cases_per_district[i][:2])
        seed(data, 2)
        assert data["agent"].is_infected.sum() == 12 + 10 + 10
        for i, district in enumerate(people_per_district):
            infected = 0
            for person_id in people_per_district[district]:
                if data["agent"].is_infected[person_id]:
                    infected += 1
            assert infected == sum(cases_per_district[i])

