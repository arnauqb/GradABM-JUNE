from pytest import fixture

from torch_june.policies.interaction_policies import SocialDistancing


class TestSocialDistancing:
    @fixture(name="sd")
    def make_sd(self):
        return SocialDistancing(
            start_date="2022-03-01",
            end_date="2022-03-15",
            beta_factors={"school": 0.3, "company": 0.5},
        )

    def test__beta_reduction(self, sd):
        assert sd.apply(beta=2, name="company") == 1
        assert sd.apply(beta=3, name="school") == 3 * 0.3
        assert sd.apply(beta=3, name="leisure") == 3
