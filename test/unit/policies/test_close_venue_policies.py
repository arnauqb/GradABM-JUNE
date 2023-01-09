from pytest import fixture
import torch
import numpy as np
import datetime

from grad_june.policies import CloseVenue, CloseVenuePolicies, Policies
from grad_june.timer import Timer
from grad_june.infection_networks.base import (
    CompanyNetwork,
    SchoolNetwork,
    InfectionNetworks,
)


class TestCloseVenue:
    @fixture(name="networks")
    def make_networks(self):
        cn = CompanyNetwork(log_beta=3.0)
        return InfectionNetworks(company=cn)

    def test__close_one_venue(self):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company",),),
            weekend_activities=(("company",),),
        )
        policy = CloseVenue(
            names=("company",),
            start_date="2022-02-01",
            end_date="2022-02-05",
            device="cpu",
        )
        while timer.date < timer.final_date:
            edges = policy.apply(
                timer=timer, edge_types=["company", "school"]
            )
            if timer.date < datetime.datetime(2022, 2, 5):
                assert edges == ["school"]
            else:
                assert edges == ["company", "school"]
            next(timer)

    def test__integration(self, inf_data, networks):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company",),),
            weekend_activities=(("company",),),
        )
        inf_data["agent"]["transmission"] = inf_data["agent"]["transmission"] + 1.0
        ret = networks(data=inf_data, timer=timer, policies=Policies([]))
        assert np.isclose(
            ret.sum().detach(), 10.0
        )  # Only the 10 in the seed don't get infected.
        policy = CloseVenue(
            names=("company",),
            start_date="2022-02-01",
            end_date="2022-02-05",
            device="cpu",
        )
        policies = Policies.from_policy_list([policy])
        ret = networks(data=inf_data, timer=timer, policies=policies)
        n_agents = inf_data["agent"].id.shape[0]
        assert np.isclose(ret.sum().detach(), n_agents)  # No-one gets infected
