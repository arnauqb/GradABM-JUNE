from pytest import fixture
import numpy as np
import torch
import torch_geometric.transforms as T

from torch_june import TorchJune, GraphLoader, AgentDataLoader, Timer
from torch_geometric.data import HeteroData


class TestModel:
    @fixture(name="model")
    def make_model(self):
        beta_priors = {
            "log_beta_company": torch.tensor(1.0, requires_grad=True),
            "log_beta_school": torch.tensor(np.log10(20.0), requires_grad=True),
            "log_beta_household": torch.tensor(np.log10(30.0), requires_grad=True),
            "log_beta_leisure": torch.tensor(1.0, requires_grad=True),
        }
        model = TorchJune(**beta_priors)
        return model

    # def test__parameters(self, model):
    #    parameters = list(model.parameters())
    #    print(parameters[0].data)
    #    print(parameters[1])
    #    assert len(parameters) == 4
    #    assert np.isclose(10 ** parameters[0].data, 10.0)
    #    assert np.isclose(10 ** parameters[1].data, 20.0)
    #    assert np.isclose(10 ** parameters[2].data, 30.0)
    #    assert np.isclose(10 ** parameters[3].data, 10.0)

    def test__run_model(self, model, inf_data, timer):
        # let transmission advance
        while timer.now < 5:
            next(timer)
        results = model(timer=timer, data=inf_data)
        # check at least someone infected
        assert results["agent"]["is_infected"].sum() > 10
        assert results["agent"]["susceptibility"].sum() < 90

    def test__model_gradient(self, model, inf_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school", "leisure", "household"),),
        )
        results = model(timer=timer, data=inf_data)
        cases = results["agent"]["is_infected"].sum()
        assert cases > 0
        loss_fn = torch.nn.MSELoss()
        random_cases = torch.rand(1)[0]
        loss = loss_fn(cases, random_cases)
        loss.backward()
        parameters = [
            getattr(model.infection_passing, "log_beta_" + at)
            for at in ["company", "school", "household", "leisure"]
        ]
        for param in parameters:
            gradient = param.grad
            assert gradient is not None
            assert gradient != 0.0

    def setup_inf_data(self, data):
        data["school"].id = torch.tensor([0])
        data["school"].people = torch.tensor([50])
        data["company"].id = torch.tensor([0])
        data["company"].people = torch.tensor([50])
        data["agent", "attends_school", "school"].edge_index = torch.vstack(
            (torch.arange(0, 50), torch.zeros(50, dtype=torch.long))
        )
        data["agent", "attends_company", "company"].edge_index = torch.vstack(
            (torch.arange(50, 100), torch.zeros(50, dtype=torch.long))
        )
        data = T.ToUndirected()(data)
        # infect some people
        susc = data["agent"]["susceptibility"].numpy()
        susc[0:100:10] = 0.0
        is_inf = data["agent"]["is_infected"].numpy()
        is_inf[0:100:10] = 1.0
        inf_t = data["agent"]["infection_time"].numpy()
        inf_t[0:100:10] = 0.0
        data["agent"].susceptibility = torch.tensor(susc)
        data["agent"].is_infected = torch.tensor(is_inf)
        data["agent"].infection_time = torch.tensor(inf_t)
        return data, is_inf

    def test__individual_gradients_companies(self, model, agent_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        # create decoupled companies and schools
        # 50 / 50
        data, is_inf = self.setup_inf_data(agent_data)

        # run
        results = model(timer=timer, data=data)
        cases = results["agent"]["is_infected"]
        assert cases.sum() > 0

        # Find person who got infected in school
        k = 0
        for i in range(50):
            if cases[i] == 1.0:
                if is_inf[i] == 1.0:  # not infected in the seed.
                    continue
                k = i
                break
        assert cases[k] == 1.0

        cases[k].backward(retain_graph=True)
        parameters = [
            getattr(model.infection_passing, "log_beta_" + at)
            for at in ["company", "school", "household", "leisure"]
        ]
        grads = np.array([p.grad.cpu() for p in parameters if p.grad is not None])
        assert len(grads) == 2
        assert grads[0] == 0.0
        assert grads[1] != 0.0

    def test__individual_gradients_schools(self, model, agent_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        # create decoupled companies and schools
        # 50 / 50
        data, is_inf = self.setup_inf_data(agent_data)

        # run
        results = model(timer=timer, data=data)
        cases = results["agent"]["is_infected"]
        assert cases.sum() > 0

        # Find person who got infected at woork
        k = 50
        reached = False
        for i in range(50, 100):
            if cases[i] == 1.0:
                if is_inf[i] == 1.0:  # not infected in the seed.
                    continue
                k = i
                reached = True
                break
        assert reached
        cases[k].backward(retain_graph=True)
        parameters = [
            getattr(model.infection_passing, "log_beta_" + at)
            for at in ["company", "school", "household", "leisure"]
        ]
        grads = np.array([p.grad.cpu() for p in parameters if p.grad is not None])
        assert len(grads) == 2
        assert grads[0] != 0.0
        assert grads[1] == 0.0

    def test__likelihood_gradient(self, model, agent_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        data, _ = self.setup_inf_data(agent_data)
        results = model(timer=timer, data=data)
        cases = results["agent"]["is_infected"]
        true_cases = 10 * torch.rand(1)
        log_likelihood = (
            torch.distributions.Normal(cases, torch.ones(1)).log_prob(true_cases).sum()
        )
        log_likelihood.backward()
        parameters = [
            getattr(model.infection_passing, "log_beta_" + at)
            for at in ["company", "school", "household", "leisure"]
        ]
        grads = np.array([p.grad.cpu() for p in parameters if p.grad is not None])
        assert len(grads) == 2
        assert grads[0] != 0.0
        assert grads[1] != 0.0
