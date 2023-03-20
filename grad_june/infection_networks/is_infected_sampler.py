import torch


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
    def forward(self, not_infected_probs, time_step):
        """
        Here we need to sample the infection status of each agent and the variant that
        the agent gets in case of infection.
        To do this, we construct a tensor of size [M+1, N], where M is the number of
        variants and N the number of agents. The extra dimension in M will represent
        the agent not getting infected, so that it can be sampled as an outcome using
        the Gumbel-Softmax reparametrization of the categorical distribution.
        """
        # calculate prob infected by any variant.
        not_infected_probs = not_infected_probs.reshape(1,-1)
        not_infected_total = torch.prod(not_infected_probs, 0)
        logits = torch.log(
            torch.vstack((not_infected_total, 1.0 - not_infected_probs))
        )
        infection = torch.nn.functional.gumbel_softmax(
            logits, dim=0, tau=0.1, hard=True
        )
        is_infected = 1.0 - infection[0, :]
        #infection_type = torch.argmax(infection[1:, :], dim=0)
        return is_infected#, infection_type
