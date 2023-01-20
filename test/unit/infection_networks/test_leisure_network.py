import pytest
import torch
import numpy as np
from torch_geometric.data import HeteroData

from grad_june.infection_networks.leisure_network import LeisureNetwork
from grad_june.policies import Policies
from grad_june.timer import Timer
import torch_geometric.transforms as T


class TestLeisureNetwork:
    @pytest.fixture(name="data")
    def make_data(self):
        data = HeteroData()
        data["agent"].id = torch.arange(0, 5)
        data["agent"].age = torch.tensor([1, 60, 20, 30, 50])
        data["agent"].sex = torch.tensor([0, 1, 1, 0, 0])
        data["agent"].susceptibility = 0.5 * torch.ones(5)
        data["agent"].transmission = 2.0 * torch.ones(5)
        data["leisure"].id = torch.arange(0, 3)
        data["agent", "attends_leisure", "leisure"].edge_index = torch.tensor(
            [[0, 1, 2], [0, 0, 0]]
        )
        data = T.ToUndirected()(data)
        return data

    @pytest.fixture(name="leisure_probabilities")
    def make_probs(self):
        return {
            "weekday": {"male": {"0-50": 0.5, "50-100": 0.2}, "female": {"0-100": 0.5}},
            "weekend": {"male": {"0-100": 1.0}, "female": {"0-100": 1.0}},
        }

    @pytest.fixture(name="ln")
    def make_ln(self, leisure_probabilities):
        return LeisureNetwork(
            log_beta=0.0, device="cpu", leisure_probabilities=leisure_probabilities
        )

    def test__parse_probs(self, ln, data):
        ln.initialize_leisure_probabilities(data)
        assert (
            ln.weekday_probabilities == torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2])
        ).all()
        assert (
            ln.weekend_probabilities == torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        ).all()

    def test__get_edge_index(self, ln, data):
        assert (
            ln._get_edge_index(data)
            == data["agent", "attends_leisure", "leisure"].edge_index
        ).all()
        assert (
            ln._get_reverse_edge_index(data)
            == data["leisure", "rev_attends_leisure", "agent"].edge_index
        ).all()

    def test__leisure_probs(self, ln, data):
        ln.initialize_leisure_probabilities(data)
        timer = Timer(initial_day="2022-05-20")
        policies = Policies()
        susc = ln._get_susceptibilities(data=data, policies=policies, timer=timer)
        assert (susc == 0.5 * torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2])).all()
        trans = ln._get_transmissions(data=data, policies=policies, timer=timer)
        assert (trans == 2.0 * torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2])).all()
        next(timer)
        susc = ln._get_susceptibilities(data=data, policies=policies, timer=timer)
        assert (susc == 0.5 * torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2])).all()
        trans = ln._get_transmissions(data=data, policies=policies, timer=timer)
        assert (trans == 2.0 * torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2])).all()
        next(timer)
        susc = ln._get_susceptibilities(data=data, policies=policies, timer=timer)
        assert (susc == 0.5 * torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])).all()
        trans = ln._get_transmissions(data=data, policies=policies, timer=timer)
        assert (trans == 2.0 * torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])).all()
