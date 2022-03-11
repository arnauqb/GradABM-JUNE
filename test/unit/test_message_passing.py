import pytest
import torch
import numpy as np
from pytest import fixture
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from torch_june import InfectionPassing


class TestInfectionPassing:
    @fixture(name="data")
    def make_graph(self):
        data = HeteroData()
        data["agent"].id = torch.arange(6)
        data["agent"].transmission = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        data["agent"].susceptibility = torch.tensor([1, 2, 3, 0.5, 0.7, 1.0])

        data["school"].id = torch.arange(2)

        edges_1 = torch.arange(6)
        edges_2 = torch.tensor([0, 0, 0, 1, 1, 1])
        data["agent", "attends_school", "school"].edge_index = torch.vstack(
            (edges_1, edges_2)
        )
        data = T.ToUndirected()(data)
        return data

    def test__infection_passing(self, data):
        inf_pass = InfectionPassing()
        infection_probabilities = inf_pass(
            data,
            edge_types=("attends_school",),
            betas={"school": 2.0},
            transmissions=data["agent"].transmission,
            susceptibilities=data["agent"].susceptibility,
        )
        expected = np.exp(-np.array([1.2, 2.4, 3.6, 1.5, 2.1, 3]))
        assert np.allclose(infection_probabilities.detach().numpy(), expected)

    def test__sample_infected(self):
        inf_pass = InfectionPassing()
        probs = torch.tensor([0.3])
        n = 1000
        ret = 0
        for _ in range(n):
            ret += inf_pass.sample_infected(probs)
        ret = ret / n
        assert np.isclose(ret, 0.7, rtol=1e-1)

        probs = torch.tensor([0.2, 0.5, 0.7, 0.3])
        n = 1000
        ret = torch.zeros(4)
        for _ in range(n):
            ret += inf_pass.sample_infected(probs)

        ret = ret / n
        assert np.allclose(ret, 1.0 - probs, rtol=1e-1)
