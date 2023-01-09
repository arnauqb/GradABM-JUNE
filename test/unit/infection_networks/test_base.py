import torch
import numpy as np
from pytest import fixture
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from grad_june.infection_networks import InfectionNetworks, IsInfectedSampler
from grad_june.infection_networks.base import (
    SchoolNetwork,
)
from grad_june.policies import Policies


class TestInfectionNetworks:
    @fixture(name="networks")
    def make_ip(self):
        sn = SchoolNetwork(log_beta=np.log10(2.0))
        return InfectionNetworks(school=sn)

    @fixture(name="small_data")
    def make_small_data(self):
        data = HeteroData()
        data["agent"].id = torch.arange(6)
        data["agent"].transmission = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        data["agent"].susceptibility = torch.tensor([1, 2, 3, 0.5, 0.7, 1.0])

        data["school"].id = torch.arange(2)
        data["school"].people = torch.tensor([2, 2])

        edges_1 = torch.arange(6)
        edges_2 = torch.tensor([0, 0, 0, 1, 1, 1])
        data["agent", "attends_school", "school"].edge_index = torch.vstack(
            (edges_1, edges_2)
        )
        data = T.ToUndirected()(data)
        return data

    def test__infection_passing(self, networks, small_data, school_timer):
        infection_probabilities = networks(
            data=small_data, timer=school_timer, policies=Policies()
        )
        expected = np.exp(-np.array([1.2, 2.4, 3.6, 1.5, 2.1, 3]))
        assert np.allclose(infection_probabilities.detach().numpy(), expected)


# def test__people_only_active_once(self, timer, inf_data):
#     data = inf_data
#     initially_infected = data["agent"].is_infected.sum()
#     # let transmission advance
#     while timer.now < 3:
#         next(timer)
#     assert timer.day_of_week == "Friday"
#     assert timer.now == 3
#     trans_updater = TransmissionUpdater()
#     data["agent"].transmission = trans_updater(data=data, timer=timer)
#     assert data["agent"].transmission.sum() > 0
#     # People that go to schools should not be infected.
#     inf_pass = InfectionPassing(
#         log_beta_school=torch.tensor(0.0),
#         log_beta_company=torch.tensor(10.0),
#         log_beta_household=torch.tensor(2.0),
#         log_beta_leisure=torch.tensor(20.0),
#     )
#     not_inf_probs = inf_pass(data=data, timer=timer).detach()
#     assert np.allclose(not_inf_probs, np.ones(len(not_inf_probs)))
#     next(timer)

#     # People that go to leisure should all be infected.
#     inf_pass = InfectionPassing(
#         log_beta_school=torch.tensor(0.0),
#         log_beta_company=torch.tensor(10.0),
#         log_beta_household=torch.tensor(20.0),
#         log_beta_leisure=torch.tensor(20000.0),
#     )
#     not_inf_probs = inf_pass(data=data, timer=timer)
#     # only not infected should be the ones already infected
#     assert np.isclose(not_inf_probs.sum(), initially_infected)
