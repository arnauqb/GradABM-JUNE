from pytest import fixture
import torch
import numpy as np
import datetime

from torch_june.policies import CloseVenue, CloseVenuePolicies
from torch_june.timer import Timer
from torch_june.message_passing import InfectionPassing


class TestCloseVenue:
    def test__close_one_venue(self):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company", ),),
            weekend_activities=(("company", ),),
        )
        policy = CloseVenue(
            name="company", start_date="2022-02-01", end_date="2022-02-05"
        )
        while timer.date < timer.final_date:
            edges = policy.apply(timer=timer, edge_types=["attends_company", "attends_school"])
            if timer.date < datetime.datetime(2022,2,5):
                assert edges == ["attends_school"]
            else:
                assert edges == ["attends_company", "attends_school"]
            next(timer)

    def test__integration(self, inf_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company", ),),
            weekend_activities=(("company", ),),
        )
        inf_data["agent"]["transmission"] = inf_data["agent"]["transmission"] + 1.0
        mp = InfectionPassing(log_beta_company = torch.tensor(3.0))
        ret = mp(data=inf_data, timer=timer)
        assert np.isclose(ret.sum().detach(), 10.0) # Only the 10 in the seed don't get infected.
        policy = CloseVenue(
            name="company", start_date="2022-02-01", end_date="2022-02-05"
        )
        policies = CloseVenuePolicies([policy])
        ret = mp(data=inf_data, timer=timer, close_venue_policies = policies)
        n_agents = inf_data["agent"].id.shape[0]
        assert np.isclose(ret.sum().detach(), n_agents) # No-one gets infected

