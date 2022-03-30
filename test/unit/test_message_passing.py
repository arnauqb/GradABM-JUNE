import pytest
import torch
import numpy as np
from pytest import fixture
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from torch_june import InfectionPassing, IsInfectedSampler, InfectionUpdater


class TestInfectionPassing:
    @fixture(name="beta_priors")
    def make_b(self):
        return {
            "school": torch.log10(torch.tensor(2.0)),
        }

    @fixture(name="inf_pass")
    def make_ip(self, beta_priors):
        return InfectionPassing(beta_school=torch.tensor(2.0))

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

    def test__get_edge_types_from_timer(self, timer, inf_pass):
        assert timer.day_of_week == "Tuesday"
        assert set(inf_pass._get_edge_types_from_timer(timer)) == set(
            [
                "attends_company",
                "attends_school",
                "attends_household",
            ]
        )
        next(timer)
        assert inf_pass._get_edge_types_from_timer(timer) == [
            "attends_leisure",
            "attends_household",
        ]
        next(timer)
        assert inf_pass._get_edge_types_from_timer(timer) == [
            "attends_household",
        ]
        while not timer.is_weekend:
            next(timer)
        assert timer.is_weekend
        assert inf_pass._get_edge_types_from_timer(timer) == [
            "attends_leisure",
        ]
        next(timer)
        assert inf_pass._get_edge_types_from_timer(timer) == [
            "attends_household",
        ]

    def test__activity_hierarchy(self, inf_pass):
        activities = [
            "attends_household",
            "attends_company",
            "attends_leisure",
            "attends_school",
        ]
        sorted = inf_pass._apply_activity_hierarchy(activities)
        assert sorted == [
            "attends_school",
            "attends_company",
            "attends_leisure",
            "attends_household",
        ]

    def test__infection_passing(self, inf_pass, small_data, school_timer):
        infection_probabilities = inf_pass(data=small_data, timer=school_timer)
        expected = np.exp(-np.array([1.2, 2.4, 3.6, 1.5, 2.1, 3]))
        assert np.allclose(infection_probabilities.detach().numpy(), expected)

    def test__sample_infected(self):
        sampler = IsInfectedSampler()
        probs = torch.tensor([0.3])
        n = 2000
        ret = 0
        for _ in range(n):
            ret += sampler(probs)
        ret = ret / n
        assert np.isclose(ret, 0.7, rtol=1e-1)

        probs = torch.tensor([0.2, 0.5, 0.7, 0.3])
        n = 2000
        ret = torch.zeros(4)
        for _ in range(n):
            ret += sampler(probs)

        ret = ret / n
        assert np.allclose(ret, 1.0 - probs, rtol=1e-1)

    def test__people_only_active_once(self, timer, inf_data):
        data = inf_data
        initially_infected = data["agent"].is_infected.sum()
        # let transmission advance
        while timer.now < 3:
            next(timer)
        assert timer.day_of_week == "Friday"
        assert timer.now == 3
        inf_updater = InfectionUpdater()
        data["agent"].transmission = inf_updater(data=data, timer=timer)
        assert data["agent"].transmission.sum() > 0
        # People that go to schools should not be infected.
        inf_pass = InfectionPassing(
            beta_school=torch.tensor(0.0),
            beta_company=torch.tensor(10.0),
            beta_household=torch.tensor(2.0),
            beta_leisure=torch.tensor(20.0),
        )
        not_inf_probs = inf_pass(data=data, timer=timer)
        assert np.allclose(not_inf_probs, np.ones(len(not_inf_probs)))
        next(timer)

        # People that go to leisure should all be infected.
        inf_pass = InfectionPassing(
            beta_school=torch.tensor(0.0),
            beta_company=torch.tensor(10.0),
            beta_household=torch.tensor(20.0),
            beta_leisure=torch.tensor(20000.0),
        )
        not_inf_probs = inf_pass(data=data, timer=timer)
        # only not infected should be the ones already infected
        assert np.isclose(not_inf_probs.sum(), initially_infected)
