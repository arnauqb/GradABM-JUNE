import torch

from torch_june.policies import Policy, PolicyCollection


class InteractionPolicy(Policy):
    spec = "interaction"


class InteractionPolicies(PolicyCollection):
    def apply(self, beta, name, timer):
        for policy in self.policies:
            beta = policy.apply(beta=beta, name=name, timer=timer)
        return beta


class SocialDistancing(InteractionPolicy):
    def __init__(self, start_date, end_date, beta_factors, device):
        super().__init__(start_date=start_date, end_date=end_date, device=device)
        beta_factors_ = {}
        for key in beta_factors:
            beta_factors_[key] = torch.tensor(float(beta_factors[key]), device=device)
        self.beta_factors = beta_factors_

    def to_device(self, device):
        for key in self.beta_factors:
            self.beta_factors[key] = self.beta_factors[key].to(device)

    def apply(self, beta, name, timer):
        if self.is_active(timer.date):
            factor = self.beta_factors.get(name, self.beta_factors.get("all", torch.tensor(1.0)))
            return beta * factor
        else:
            return beta
