from pytest import fixture
import numpy as np
import torch
import torch_geometric.transforms as T

from torch_june import TorchJune, Timer
from torch_june.infection_networks import (
    CompanyNetwork,
    SchoolNetwork,
    HouseholdNetwork,
    InfectionNetworks,
)


class TestModel:
    @fixture(name="model")
    def make_model(self):
        cn = CompanyNetwork(log_beta=0.5)
        hn = HouseholdNetwork(log_beta=0.5)
        sn = SchoolNetwork(log_beta=0.5)
        networks = InfectionNetworks(household=hn, company=cn, school=sn)
        model = TorchJune(infection_networks=networks)
        return model

    def test__run_model(self, model, inf_data, timer):
        # let transmission advance
        while timer.now < 3:
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
            weekday_activities=(("company", "school", "household"),),
        )
        results = model(timer=timer, data=inf_data)
        cases = results["agent"]["is_infected"].sum()
        assert cases > 0
        loss_fn = torch.nn.MSELoss()
        random_cases = torch.rand(1)[0]
        loss = loss_fn(cases, random_cases)
        loss.backward()
        beta = model.infection_networks["company"].log_beta
        assert beta.grad is not None
        assert beta.grad != 0
        beta = model.infection_networks["school"].log_beta
        assert beta.grad is not None
        assert beta.grad != 0

    @fixture(name="data2")
    def setup_data(self, inf_data):
        data = inf_data
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
        # delete reverse before rebuilding it.
        del data["rev_attends_school"]
        del data["rev_attends_company"]
        del data["rev_attends_household"]
        data = T.ToUndirected()(data)
        is_inf = data["agent"].is_infected.numpy()
        return data, is_inf

    def test__individual_gradients_schools(self, model, data2):
        data, is_inf = data2
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        for i in range(4):  # make sure someone gets infected.
            data = model(timer=timer, data=data)
        cases = data["agent"]["is_infected"]
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
        p_company = model.infection_networks["company"].log_beta
        p_school = model.infection_networks["school"].log_beta
        company_grad = p_company.grad.cpu()
        school_grad = p_school.grad.cpu()
        assert school_grad != 0.0
        assert company_grad == 0.0

    def test__individual_gradients_companies(self, model, data2):
        data, is_inf = data2
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company", "school"),),
            weekend_activities=(("company", "school"),),
        )
        # create decoupled companies and schools
        # 50 / 50
        # run
        for i in range(3):
            results = model(timer=timer, data=data)
            next(timer)
        cases = results["agent"]["is_infected"]
        assert cases.sum() > 10

        # Find person who got infected at woork
        k = 50
        reached = False
        for i in range(50, 100):
            if cases[i] == 1.0:
                if is_inf[i] == 1.0:  # not infected in the seed.
                    continue
                k = i
                reached = True
                # break
        assert reached
        cases[k].backward(retain_graph=True)
        p_company = model.infection_networks["company"].log_beta
        p_school = model.infection_networks["school"].log_beta
        company_grad = p_company.grad.cpu()
        school_grad = p_school.grad.cpu()
        assert school_grad == 0
        assert company_grad != 0

    def test__likelihood_gradient(self, model, data2):
        data, is_inf = data2
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        results = model(timer=timer, data=data)
        cases = results["agent"]["is_infected"]
        true_cases = 10 * torch.rand(1)
        log_likelihood = (
            torch.distributions.Normal(cases, torch.ones(1)).log_prob(true_cases).sum()
        )
        log_likelihood.backward()
        parameters = [
            model.infection_networks[name].log_beta
            for name in ["company", "school"]
        ]
        grads = np.array([p.grad.cpu() for p in parameters])
        assert len(grads) == 2
        assert grads[0] != 0.0
        assert grads[1] != 0.0

    def test__symptoms_update(self, model, data2):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekday_activities=(("company", "school"),),
        )
        data, _ = data2
        results = model(timer=timer, data=data)
        is_infected = results["agent"].is_infected
        cases = int(is_infected.sum().item())
        assert cases > 10
        symptoms = data["agent"].symptoms
        assert (
            symptoms["current_stage"][is_infected.bool()]
            == 2 * torch.ones(cases, dtype=torch.long)
        ).all()
