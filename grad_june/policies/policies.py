from abc import ABC
import yaml
import re
import datetime
import torch

import grad_june
from grad_june.utils import read_date
from grad_june.paths import default_config_path


class Policy(torch.nn.Module):
    def __init__(self, start_date, end_date, device):
        super().__init__()
        self.start_date = read_date(start_date)
        self.end_date = read_date(end_date)
        self.device = device

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


class PolicyCollection(torch.nn.Module):
    def __init__(self, policies: Policy):
        """
        A collection of like policies active on the same date
        """
        super().__init__()
        self.policies = torch.nn.ModuleList(policies)

    def __getitem__(self, idx):
        return self.policies[idx]


class Policies(torch.nn.Module):
    def __init__(
        self,
        interaction_policies=None,
        quarantine_policies=None,
        close_venue_policies=None,
    ):
        super().__init__()
        self.interaction_policies = interaction_policies
        self.quarantine_policies = quarantine_policies
        self.close_venue_policies = close_venue_policies

    @classmethod
    def from_policy_list(cls, policies):
        if policies is None:
            policies = torch.nn.ModuleList([])
        from grad_june.policies import (
            InteractionPolicies,
            QuarantinePolicies,
            CloseVenuePolicies,
        )

        interaction_policies = InteractionPolicies(
            cls._get_policies_by_type(policies, "interaction")
        )
        quarantine_policies = QuarantinePolicies(
            cls._get_policies_by_type(policies, "quarantine")
        )
        close_venue_policies = CloseVenuePolicies(
            cls._get_policies_by_type(policies, "close_venue")
        )
        return cls(
            interaction_policies=interaction_policies,
            quarantine_policies=quarantine_policies,
            close_venue_policies=close_venue_policies,
        )

    @classmethod
    def from_file(cls, fpath=default_config_path):
        with open(fpath, "r") as f:
            params = yaml.safe_load(f)
        return cls.from_parameters(params)

    @classmethod
    def from_parameters(cls, params):
        policy_params = params.get("policies", {})
        device = params["system"]["device"]
        policies = []
        for policy_collection in policy_params.values():
            for policy_name, policy_config in policy_collection.items():
                policies += cls._parse_policy_config(
                    policy_config, name=policy_name, device=device
                )
        return cls.from_policy_list(policies)

    @staticmethod
    def _parse_policy_config(config, name, device):
        camel_case_key = "".join(x.capitalize() or "_" for x in name.split("_"))
        policies = []
        policy_class = getattr(grad_june.policies, camel_case_key)
        if "start_date" not in config:
            for policy_i, policy_data_i in config.items():
                if (
                    "start_date" not in policy_data_i.keys()
                    or "end_date" not in policy_data_i.keys()
                ):
                    raise ValueError("policy config file not valid.")
                policies.append(policy_class(**policy_data_i, device=device))
        else:
            policies.append(policy_class(**config, device=device))
        return policies

    @classmethod
    def _get_policies_by_type(cls, policies, type):
        return [policy for policy in policies if policy.spec == type]

    def apply(self, data, timer):
        if self.quarantine_policies:
            self.quarantine_policies.apply(
                timer=timer, symptom_stages=data["agent"]["symptoms"]["current_stage"]
            )
