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
        #print(f"------{timer.now}------")
        time_from_infection = timer.now - data["agent"].infection_time
        #print(time_from_infection)
        #print(data["agent"].is_infected)
        sign = (torch.sign(time_from_infection - shift + 1e-10) + 1) / 2
        #print("sign")
        #print(sign)
        #print(sign)
        aux = torch.exp(-torch.lgamma(shape)) * torch.pow(
            (time_from_infection - shift) * rate, shape - 1.0
        )
        #print(aux)
        aux2 = torch.exp((shift - time_from_infection) * rate) * rate
        #print(aux2)
        ret = max_infectiousness * sign * aux * aux2 * data["agent"].is_infected
        #print("ret")
        #print(ret)
        return ret
