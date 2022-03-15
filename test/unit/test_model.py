from pytest import fixture
import numpy as np
import torch

from torch_june import TorchJune, GraphLoader, AgentDataLoader, Timer
from torch_june.infections import Infections
from torch_geometric.data import HeteroData


class TestModel:
    @fixture(name="data")
    def make_data(self, june_world_path):
        data = HeteroData()
        data = GraphLoader(june_world_path, k_leisure=1).load_graph(data)
        AgentDataLoader(june_world_path).load_agent_data(data)
        return data

    @fixture(name="model")
    def make_model(self, data, sampler):
        betas = {"company": 1.0, "school": 2.0, "household": 3.0, "leisure": 1.0}
        initial_infected = np.zeros(len(data["agent"]["id"]))
        initial_infected[0] = 1
        initial_infected = torch.tensor(initial_infected, requires_grad=True)
        infections = Infections(
            sampler(len(data["agent"]["id"])), initial_infected=initial_infected
        )
        model = TorchJune(data=data, betas=betas, infections=infections)
        return model

    @fixture(name="trans_susc")
    def make_trans_suc(self):
        trans = np.zeros(6640)
        trans[0] = 0.2
        susc = np.ones(6640)
        susc[0] = 0.0
        return torch.tensor(trans), torch.tensor(susc)

    @fixture(name="timer")
    def make_timer(self):
        timer = Timer(
            initial_day="2022-03-18",
            total_days=10,
            weekday_step_duration=(8, 8, 8),
            weekend_step_duration=(
                12,
                12,
            ),
            weekday_activities=(
                ("company", "school"),
                ("leisure",),
                ("household",),
            ),
            weekend_activities=(("leisure",), ("household",)),
        )
        return timer

    def test__get_edge_types_from_timer(self, model, timer):
        assert set(model._get_edge_types_from_timer(timer)) == set(
            [
                "attends_company",
                "attends_school",
            ]
        )
        next(timer)
        assert model._get_edge_types_from_timer(timer) == [
            "attends_leisure",
        ]
        next(timer)
        assert model._get_edge_types_from_timer(timer) == [
            "attends_household",
        ]
        next(timer)
        assert timer.is_weekend
        assert model._get_edge_types_from_timer(timer) == [
            "attends_leisure",
        ]
        next(timer)
        assert model._get_edge_types_from_timer(timer) == [
            "attends_household",
        ]

    def test__parameters(self, model):
        parameters = list(model.parameters())[0].data
        assert len(parameters) == 4
        assert parameters[0].data == 1.0
        assert parameters[1].data == 2.0
        assert parameters[2].data == 3.0
        assert parameters[3].data == 1.0

    def test__run_model(self, model, trans_susc, timer):
        trans, susc = trans_susc
        results = model(timer=timer, susceptibilities=susc)
        assert results.shape == (26, 6640)

    def test__model_gradient(self, model, trans_susc, timer):
        trans, susc = trans_susc
        results = model(timer=timer, susceptibilities=susc)
        daily_cases = torch.sum(results, dim=1)
        assert len(daily_cases) == 26
        assert sum(daily_cases) > 0.0

        loss_fn = torch.nn.MSELoss()
        random_cases = 1e7 * torch.ones(26, dtype=torch.float64)
        loss = loss_fn(daily_cases, random_cases)
        loss.backward()
        parameters = [p for p in model.parameters()][0]
        gradient = parameters.grad
        for v in gradient:
            assert v is not None
        assert sum(gradient) != 0.0
