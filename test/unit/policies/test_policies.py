import datetime
import pytest

from torch_june.policies import Policy, Policies
from torch_june.policies.interaction_policies import (
    InteractionPolicies,
    SocialDistancing,
)


class TestPolicies:
    def test__policy(self):
        policy = Policy(start_date="2022-02-01", end_date="2022-02-05")
        assert policy.start_date == datetime.datetime(2022, 2, 1)
        assert policy.end_date == datetime.datetime(2022, 2, 5)
        with pytest.raises(NotImplementedError) as exc_info:
            policy.apply()

    def test__p_collection(self):
        policies = Policies(
            [
                SocialDistancing(
                    start_date="2022-02-01",
                    end_date="2022-02-05",
                    beta_factors={"school": 0.5},
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
        assert policies.interaction_policies[0].beta_factors == {
            "school": 0.5,
            "company": 0.5,
        }
        assert policies.interaction_policies[1].start_date == datetime.datetime(
            2022, 3, 15
        )
        assert policies.interaction_policies[1].end_date == datetime.datetime(
            2022, 4, 15
        )
        assert policies.interaction_policies[1].beta_factors == {
            "leisure": 0.5,
        }
