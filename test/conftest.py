from pathlib import Path

from pytest import fixture

@fixture(scope="session", name="june_world_path")
def get_june_world_path():
    return Path(__file__).parent / "data/june_world.hdf5"

