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
        policies = Policies.from_parameters()
        assert policies.interaction_policies[0].start_date == datetime.datetime(
            2020, 3, 16
        )
        assert policies.interaction_policies[0].end_date == datetime.datetime(
            2020, 3, 24
        )
        assert policies.interaction_policies[0].beta_factors == {
            "leisure": 0.65,
            "care_home": 0.65,
            "school": 0.65,
            "university": 0.65,
            "company": 0.65,
        }
        assert policies.interaction_policies[1].start_date == datetime.datetime(
            2020, 3, 24
        )
        assert policies.interaction_policies[1].end_date == datetime.datetime(
            2020, 5, 11
        )
        assert policies.interaction_policies[1].beta_factors == {
            "leisure": 0.45,
            "care_home": 0.45,
            "school": 0.45,
            "university": 0.45,
            "company": 0.45,
        }
        assert policies.close_venue_policies[0].start_date == datetime.datetime(
            2020, 3, 21
        )
        assert policies.close_venue_policies[0].end_date == datetime.datetime(
            2020, 7, 4
        )
        assert policies.close_venue_policies[0].edge_type_to_close == set(
            ["attends_leisure", "attends_school"]
        )
        assert policies.quarantine_policies[0].start_date == datetime.datetime(
            2020, 3, 16
        )
        assert policies.quarantine_policies[0].end_date == datetime.datetime(
            9999, 3, 24
        )
        assert policies.quarantine_policies[0].stage_threshold == 4
