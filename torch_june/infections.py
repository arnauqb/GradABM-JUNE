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


class InfectionUpdater(torch.nn.Module):
    def forward(self, data, timer):
        shape = data["agent"]["infection_parameters"]["shape"]
        shift = data["agent"]["infection_parameters"]["shift"]
        rate = data["agent"]["infection_parameters"]["rate"]
        max_infectiousness = data["agent"]["infection_parameters"]["max_infectiousness"]
        time = timer.now
        sign = (torch.sign(time - data["agent"].infection_time) + 1.0) / 2.0
        aux = torch.exp(torch.lgamma(shape)) * torch.pow(
            (time - shift) * rate, shape - 1.0
        )
        aux2 = torch.exp((shift - time) * rate) * rate
        return max_infectiousness * sign * aux * aux2 * data["agent"].is_infected
