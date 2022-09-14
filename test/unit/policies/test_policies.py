import datetime
import pytest
import torch

from torch_june.policies import Policy, Policies
from torch_june.policies.interaction_policies import (
    InteractionPolicies,
    SocialDistancing,
)


class TestPolicies:
    def test__policy(self):
        policy = Policy(start_date="2022-02-01", end_date="2022-02-05", device="cpu")
        assert policy.start_date == datetime.datetime(2022, 2, 1)
        assert policy.end_date == datetime.datetime(2022, 2, 5)
        assert policy.device == "cpu"
        with pytest.raises(NotImplementedError) as exc_info:
            policy.apply()

    def test__p_collection(self):
        policies = Policies.from_policy_list(
            [
                SocialDistancing(
                    start_date="2022-02-01",
                    end_date="2022-02-05",
                    beta_factors={"school": 0.5},
                    device="cpu",
                )
            ]
        )
        assert type(policies.interaction_policies) == InteractionPolicies

    def test__from_default_parameters(self):
        policies = Policies.from_file()
        assert policies.interaction_policies[0].start_date == datetime.datetime(
            2022, 2, 15
        )
        assert policies.interaction_policies[0].end_date == datetime.datetime(
            2022, 3, 15
        )
        assert policies.interaction_policies[0].beta_factors["school"] == 0.5
        assert policies.interaction_policies[0].beta_factors["company"] == 0.5
        assert policies.interaction_policies[1].start_date == datetime.datetime(
            2022, 3, 15
        )
        assert policies.interaction_policies[1].end_date == datetime.datetime(
            2022, 4, 15
        )
        assert policies.interaction_policies[1].beta_factors["pub"] == 0.5
        assert policies.interaction_policies[1].beta_factors["grocery"] == 0.5
        assert policies.interaction_policies[1].beta_factors["cinema"] == 0.5
        assert policies.interaction_policies[1].beta_factors["gym"] == 0.5
        assert policies.interaction_policies[1].beta_factors["visit"] == 0.5
