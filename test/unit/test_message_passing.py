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
        data["agent"].susceptibility = torch.tensor([1, 2, 3, 4, 5, 6.0])

        data["school"].id = torch.arange(2)
        data["school"].beta = 2.0 * torch.ones(2)

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
            data, edge_types=("attends_school",)
        )["attends_school"]
        expected = 1.0 - torch.exp(-torch.tensor([1.2, 2.4, 3.6, 12, 15, 18]))
        assert (infection_probabilities == expected).all()
