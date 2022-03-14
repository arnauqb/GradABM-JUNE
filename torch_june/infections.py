import torch


class InfectionSampler:
    def __init__(self, max_infectiousness, shape, rate, shift):
        self.max_infectiousness = max_infectiousness
        self.shape = shape
        self.rate = rate
        self.shift = shift

    def __call__(self, n):
        maxi = self.max_infectiousness.sample((n,))
        shape = self.shape.sample((n,))
        rate = self.rate.sample((n,))
        shift = self.shift.sample((n,))
        return torch.vstack((maxi, shape, rate, shift))


class Infections:
    def __init__(self, parameters, initial_infected=None):
        self.max_infectiousness, self.shape, self.rate, self.shift = parameters
        if initial_infected is None:
            initial_infected = torch.zeros(
                len(self.max_infectiousness), requires_grad=True
            )
        self.is_infected = initial_infected
        self.infection_times = (
            -1.0 * torch.ones(len(self.max_infectiousness), requires_grad=True)
            + initial_infected
        )

    def update(self, new_infected, infection_time):
        self.is_infected = self.is_infected + new_infected
        self.infection_times = self.infection_times + new_infected * (
            1.0 + infection_time
        )

    def get_transmissions(self, time):
        sign = (torch.sign(time - self.infection_times) + 1.0) / 2.0
        aux = torch.exp(torch.lgamma(self.shape)) * torch.pow(
            (time - self.shift) * self.rate, self.shape - 1.0
        )
        aux2 = torch.exp((self.shift - time) * self.rate) * self.rate
        return sign * aux * aux2 * self.is_infected
