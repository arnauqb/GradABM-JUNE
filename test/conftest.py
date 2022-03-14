from pathlib import Path
from torch.distributions import Normal, LogNormal
from pytest import fixture

from torch_june.infections import InfectionSampler

@fixture(scope="session", name="june_world_path")
def get_june_world_path():
    return Path(__file__).parent / "data/june_world.hdf5"

@fixture(scope="session", name="june_world_path_only_people")
def get_june_world_path_only_people():
    return Path(__file__).parent / "data/june_world_only_people.hdf5"

@fixture(scope="session", name="sampler")
def make_sampler():
    max_infectiousness = LogNormal(0, 0.5)# * 1.7
    shape = Normal(1.56, 0.08)
    rate = Normal(0.53, 0.03)
    shift = Normal(-2.12, 0.1)
    return InfectionSampler(max_infectiousness, shape, rate, shift)
