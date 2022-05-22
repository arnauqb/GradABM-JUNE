from pytest import fixture
import datetime
import torch
import numpy as np

from torch_june.policies.interaction_policies import (
    SocialDistancing,
    InteractionPolicies,
)
from torch_june.message_passing import InfectionPassing
from torch_june.timer import Timer


class TestSocialDistancing:
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
            print("---")
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

    def test__integration(self, inf_data):
        timer = Timer(
            initial_day="2022-02-01",
            total_days=10,
            weekday_step_duration=(24,),
            weekend_step_duration=(24,),
            weekday_activities=(("company",),),
            weekend_activities=(("company",),),
        )
        inf_data["agent"]["transmission"] = inf_data["agent"]["transmission"] + 1.0
        mp = InfectionPassing(log_beta_school=torch.tensor(2.0))

        sd1 = SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"school": 0.3, "company": 0.5},
            device="cpu",
        )
        int_pols = InteractionPolicies([sd1])
        ret1 = mp(data=inf_data, timer=timer, interaction_policies=int_pols)
        ts1 = -torch.log(ret1) * timer.duration
        ts1 = ts1[ts1 > 5e-6].detach().numpy()

        sd2 = SocialDistancing(
            start_date="2022-02-01",
            end_date="2022-02-05",
            beta_factors={"school": 0.6, "company": 0.2},
            device="cpu",
        )
        int_pols = InteractionPolicies([sd2])
        ret2 = mp(data=inf_data, timer=timer, interaction_policies=int_pols)
        ts2 = -torch.log(ret2) * timer.duration
        ts2 = ts2[ts2 > 5e-6].detach().numpy()
        assert np.allclose(ts2 / ts1, 0.2 / 0.5 * torch.ones(ts2.shape[0]))
