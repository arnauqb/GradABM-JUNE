from torch_june.infection_seed import infect_people_at_indices


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
