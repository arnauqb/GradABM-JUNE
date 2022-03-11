from pytest import fixture
import numpy as np
import torch

from torch_june import TorchJune, GraphLoader, AgentDataLoader
from torch_geometric.data import HeteroData


class TestModel:
    @fixture(name="model")
    def make_model(self, june_world_path):
        betas = {"company": 1.0, "school": 2.0, "household": 3.0}
        data = HeteroData()
        data = GraphLoader(june_world_path).load_graph(data)
        AgentDataLoader(june_world_path).load_agent_data(data)
        model = TorchJune(data=data, betas=betas)
        return model

    @fixture(name="trans_susc")
    def make_trans_suc(self):
        trans = np.zeros(6640)
        trans[0] = 0.2
        susc = np.ones(6640)
        susc[0] = 0.0
        return torch.tensor(trans, requires_grad=True), torch.tensor(
            susc, requires_grad=True
        )

    def test__parameters(self, model):
        parameters = list(model.parameters())[0].data
        assert len(parameters) == 3
        assert parameters[0].data == 1.0
        assert parameters[1].data == 2.0
        assert parameters[2].data == 3.0

    def test__run_model(self, model, trans_susc):
        trans, susc = trans_susc
        n_timesteps = 10
        results = model(
            n_timesteps=n_timesteps, transmissions=trans, susceptibilities=susc
        )
        assert results.shape == (10, 6640)

    def test__model_gradient(self, model, trans_susc):
        trans, susc = trans_susc
        n_timesteps = 10
        results = model(
            n_timesteps=n_timesteps, transmissions=trans, susceptibilities=susc
        )
        daily_cases = torch.sum(results, dim=1)
        assert len(daily_cases) == 10

        loss_fn = torch.nn.MSELoss()
        random_cases = 1e7 * torch.ones(10, dtype=torch.float64)
        loss = loss_fn(daily_cases, random_cases)
        loss.backward()
        parameters = [p for p in model.parameters()][0]
        gradient = parameters.grad
        for v in gradient:
            assert v is not None
            assert v != 0
