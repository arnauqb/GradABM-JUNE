from torch_june.policies import Policy

class InteractionPolicy(Policy):
    spec = "interaction"

class InteractionPolicies:
    def __init__(self, policies):
        self.policies = policies

    def apply(self, beta, name, timer):
        for policy in self.policies:
            beta = policy.apply(beta=beta, name=name, timer=timer)
        return beta


class SocialDistancing(InteractionPolicy):
    def __init__(self, start_date, end_date, beta_factors):
        super().__init__(start_date=start_date, end_date=end_date)
        self.beta_factors = beta_factors

    def apply(self, beta, name, timer):
        if self.is_active(timer.date):
            factor = self.beta_factors.get(name, 1.0)
            return beta * factor
        else:
            return beta
