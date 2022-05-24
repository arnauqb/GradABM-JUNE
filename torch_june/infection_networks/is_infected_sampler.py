import torch
from pyro.distributions import RelaxedBernoulliStraightThrough


class IsInfectedSampler(torch.nn.Module):
    def forward(self, not_infected_probs):
        infected_probs = 1.0 - not_infected_probs
        dist = RelaxedBernoulliStraightThrough(temperature=0.1, probs=infected_probs)
        return dist.rsample()
