import torch
import pyro
from pyro import distributions

class my_round_func(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class IsInfectedSampler(torch.nn.Module):
    i = 0

    def forward(self, not_infected_probs):
        infected_probs = 1.0 - not_infected_probs
        dist = distributions.RelaxedBernoulliStraightThrough(
            temperature=torch.tensor(0.1),
            probs=infected_probs,
        )
        ret = pyro.sample(f"inf_{self.i}", dist)
        #ret = dist.rsample()
        #print(ret)
        self.i += 1
        return ret

