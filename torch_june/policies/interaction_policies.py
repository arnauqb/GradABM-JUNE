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
            # beta_factors[key] = torch.tensor(float(beta_factors[key]), device=device)
            beta_factors_[key] = torch.nn.Parameter(
                torch.tensor(float(beta_factors[key]), device=device)
            )
        self.beta_factors = torch.nn.ParameterDict(beta_factors_)
        # self.beta_factors = beta_factors

    def apply(self, beta, name, timer):
        if self.is_active(timer.date):
            factor = self.beta_factors.get(name, 1.0)
            return beta * factor
        else:
            return beta
