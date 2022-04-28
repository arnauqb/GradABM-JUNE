from abc import ABC
import re
import datetime 

from torch_june.utils import read_date


class Policy(ABC):
    def __init__(self, start_date, end_date):
        self.start_date = read_date(start_date)
        self.end_date = read_date(end_date)

    def apply(self):
        raise NotImplementedError

    def is_active(self, date: datetime.datetime) -> bool:
        """
        Returns true if the policy is active, false otherwise

        Parameters
        ----------
        date:
            date to check
        """
        return self.start_date <= date < self.end_date

class Policies:
    def __init__(self, policies=None):
        if policies is None:
            policies = []
        self.policies = policies
        from torch_june.policies.interaction_policies import InteractionPolicies
        self.interaction_policies = InteractionPolicies(self._get_policies_by_type(policies, "interaction"))

    def _get_policies_by_type(self, policies, type):
        return [policy for policy in policies if policy.spec == type]


class PolicyCollection:
    def __init__(self, policies: list[Policy]):
        """
        A collection of like policies active on the same date
        """
        self.policies = policies
