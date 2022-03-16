from pytest import fixture
import numpy as np
import torch

from torch_june import TorchJune, GraphLoader, AgentDataLoader, Timer
from torch_geometric.data import HeteroData


class TestModel:
    @fixture(name="model")
    def make_model(self):
        beta_priors = {
            "company": 10.0,
            "school": 20.0,
            "household": 30.0,
            "leisure": 10.0,
        }
        model = TorchJune(beta_priors=beta_priors)
        return model

    def test__parameters(self, model):
        parameters = list(model.parameters())[0].data
        assert len(parameters) == 4
        assert parameters[0].data == 10.0
        assert parameters[1].data == 20.0
        assert parameters[2].data == 30.0
        assert parameters[3].data == 10.0

    def test__run_model(self, model, inf_data, timer):
        # let transmission advance
        while timer.now < 5:
            next(timer)
        results = model(timer=timer, data=inf_data)
        # check at least someone infected
        assert results["agent"]["is_infected"].sum() > 10
        assert results["agent"]["susceptibility"].sum() < 90

    def test__model_gradient(self, model, inf_data, timer):
        results = model(timer=timer, data=inf_data)
        cases = results["agent"]["is_infected"].sum()
        assert cases > 0
        loss_fn = torch.nn.MSELoss()
        random_cases = torch.rand(1)
        loss = loss_fn(cases, random_cases)
        loss.backward()
        parameters = [p for p in model.parameters()][0]
        gradient = parameters.grad
        assert len(gradient) > 0
        for v in gradient:
            assert v is not None
        assert sum(gradient) != 0.0
