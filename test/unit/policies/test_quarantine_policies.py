import torch
import pytest
import numpy as np

from grad_june.policies import Quarantine, QuarantinePolicies, Policies
from grad_june.timer import Timer
from grad_june.infection_networks.base import (
    CompanyNetwork,
    HouseholdNetwork,
    InfectionNetworks,
)


class TestQuarantine:
    @pytest.fixture(name="networks")
    def make_networks(self):
        cn = CompanyNetwork(log_beta=3.0)
        hn = HouseholdNetwork(log_beta=3.0)
        return InfectionNetworks(household=hn, company=cn)

    def test__match_symptoms(self):
        symptom_stages = torch.tensor([0, 1, 2, 3, 4])
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company", "household"),),
            weekend_activities=(("company", "household"),),
        )
        quarantine = Quarantine(
            stage_threshold=3,
            start_date="2022-02-01",
            end_date="2022-02-05",
            device="cpu",
        )
        quarantine_array = quarantine.apply(timer=timer, symptom_stages=symptom_stages)
        assert (quarantine_array == torch.tensor([1, 1, 1, 0, 0])).all()

    def test__integration(self, inf_data, networks):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company",),),
            weekend_activities=(("company", "household"),),
        )
        inf_data["agent"]["transmission"] = inf_data["agent"]["transmission"] + 1.0
        n_agents = inf_data["agent"].id.shape[0]
        inf_data["agent"]["symptoms"]["current_stage"] = 5 * torch.ones(n_agents)
        quarantine = Quarantine(
            stage_threshold=3,
            start_date="2022-02-01",
            end_date="2022-03-15",
            device="cpu",
        )
        policies = Policies.from_policy_list([quarantine])
        ret = networks(data=inf_data, timer=timer, policies=policies)
        for i in range(3):
            assert np.isclose(
                ret[i,:].sum().detach(), n_agents
            )  # No-one gets infected since they all quarantine
        while not timer.is_weekend:
            next(timer)
        ret = networks(
            data=inf_data,
            timer=timer,
            policies=policies,
        )
        assert np.isclose(
            ret[0,:].sum().detach().item(), 10.0
        )  # people living in the same household get infected, seed survives
