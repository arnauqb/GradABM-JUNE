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


class IsInfectedSampler(pyro.nn.PyroModule):
    i = 0

    def forward(self, not_infected_probs, time_step):
        infected_probs = 1.0 - not_infected_probs
        ret = pyro.distributions.RelaxedBernoulliStraightThrough(
            temperature=torch.tensor(0.1), probs=infected_probs
        ).rsample()
        # dist = distributions.RelaxedBernoulliStraightThrough(
        #    temperature=torch.tensor(0.1),
        #    probs=infected_probs,
        # ).to_event(1)
        # dist = distributions.Bernoulli(infected_probs).to_event(1)
        # dist = distributions.Bernoulli(infected_probs).to_event(1)
        # ret = pyro.sample(
        #    f"inf_{time_step}",
        #    dist,
        #    #infer=dict(
        #    #    baseline={"use_decaying_avg_baseline": True, "baseline_beta": 0.95}
        #    #),
        # )
        self.i += 1
        return ret
