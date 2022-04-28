from torch_june.policies import Policy

class SocialDistancing(Policy):
    def __init__(self, start_date, end_date, beta_factors):
        super().__init__(start_date=start_date, end_date=end_date)
        self.beta_factors = beta_factors

    def apply(self, beta, name):
        factor = self.beta_factors.get(name, 1.0)
        return beta * factor

