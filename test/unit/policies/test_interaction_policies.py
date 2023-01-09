from pytest import fixture
import datetime
import torch
import numpy as np

from grad_june.policies.interaction_policies import (
    SocialDistancing,
    InteractionPolicies,
)
from grad_june.timer import Timer
from grad_june.policies import Policies
from grad_june.infection_networks.base import (
    CompanyNetwork,
    SchoolNetwork,
    InfectionNetworks,
)


class TestSocialDistancing:
    @fixture(name="networks")
    def make_networks(self):
        sn = SchoolNetwork(log_beta=2.0)
        cn = CompanyNetwork(log_beta=0.0)
        return InfectionNetworks(school=sn, company=cn)

    @fixture(name="sd")
    def make_sd(self):
        return SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"school": 0.3, "company": 0.5},
            device="cpu",
        )

    def test__beta_reduction(self, sd, timer):
        while timer.date < timer.final_date:
            if timer.date < datetime.datetime(2022, 2, 5):
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="school", timer=timer).item(),
                    3 * 0.3,
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(2), name="company", timer=timer).item(),
                    1,
                )
            else:
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="school", timer=timer).item(), 3
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(2), name="company", timer=timer).item(),
                    2,
                )
            assert np.isclose(
                sd.apply(beta=torch.tensor(3), name="leisure", timer=timer).item(), 3
            )
            next(timer)

    def test__beta_reduction_all(self, timer):
        sd = SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"all": 0.8},
            device="cpu",
        )
        while timer.date < timer.final_date:
            if timer.date < datetime.datetime(2022, 2, 5):
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="school", timer=timer).item(),
                    3 * 0.8,
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(2), name="company", timer=timer).item(),
                    2 * 0.8,
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="leisure", timer=timer).item(), 3 * 0.8
                )
            else:
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="school", timer=timer).item(), 3
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(2), name="company", timer=timer).item(),
                    2,
                )
                assert np.isclose(
                    sd.apply(beta=torch.tensor(3), name="leisure", timer=timer).item(), 3 
                )
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
        sd1 = SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"school": 0.3, "company": 0.5},
            device="cpu",
        )
        policies = Policies.from_policy_list([sd1])
        ret1 = networks(data=inf_data, timer=timer, policies=policies)
        ts1 = -torch.log(ret1) * timer.duration
        ts1 = ts1[ts1 > 5e-6].detach().numpy()

        sd2 = SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"school": 0.6, "company": 0.2},
            device="cpu",
        )
        policies = Policies.from_policy_list([sd2])
        ret2 = networks(data=inf_data, timer=timer, policies=policies)
        ts2 = -torch.log(ret2) * timer.duration
        ts2 = ts2[ts2 > 5e-6].detach().numpy()
        assert np.allclose(ts2 / ts1, 0.2 / 0.5 * torch.ones(ts2.shape[0]))
