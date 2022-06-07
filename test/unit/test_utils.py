import pytest
import numpy as np
import datetime
import torch
import random
from pathlib import Path

from torch_june.utils import (
    parse_age_probabilities,
    read_date,
    fix_seed,
    read_path,
    create_simple_connected_graph,
)
from torch_june.paths import torch_june_path


class TestReadPath:
    def test__read_path(self):
        assert read_path("/test/to/path") == Path("/test/to/path")
        assert read_path("@torch_june/to/path") == torch_june_path / "to/path"


class TestParseProbabilities:
    def test__parsing(self):
        input = {"5-20": 0.2, "25-40": 0.7}
        probs = parse_age_probabilities(input)
        assert (probs[:5] == np.zeros(5)).all()
        assert (probs[5:20] == 0.2 * np.ones(15)).all()
        assert (probs[20:25] == np.zeros(5)).all()
        assert (probs[25:40] == 0.7 * np.ones(15)).all()
        assert (probs[40:] == np.zeros(60)).all()


class TestReadDate:
    def test__read(self):
        assert read_date("2022-02-01") == datetime.datetime(2022, 2, 1)
        assert read_date(datetime.datetime(2022, 2, 1)) == datetime.datetime(2022, 2, 1)
        with pytest.raises(TypeError, match="must be a string") as exc_info:
            read_date(1015)


class TestFixSeed:
    def test__fix_seed(self):
        fix_seed(1992)
        a1 = torch.rand(
            10,
        )
        b1 = np.random.rand(
            10,
        )
        c1 = random.random()
        fix_seed(1992)
        a2 = torch.rand(
            10,
        )
        b2 = np.random.rand(
            10,
        )
        c2 = random.random()
        assert (a1 == a2).all()
        assert (b1 == b2).all()
        assert c1 == c2


class TestCreateSimpleGraph:
    def test(self):
        data = create_simple_connected_graph(100)
        assert data["agent"].id.shape[0] == 100
        assert data["agent"].age.shape[0] == 100
        assert data["agent"].sex.shape[0] == 100
        #for i in range(10):
        #    assert data["agent"]["susceptibility"][i] == 0.0
        #    assert data["agent"]["is_infected"][i] == 1
        #    assert data["agent"]["infection_time"][i] == 0.0
        #    assert data["agent"]["symptoms"]["next_stage"][i] == 2.0
        #for i in range(10, 100):
        #    assert data["agent"]["susceptibility"][i] == 1.0
        #    assert data["agent"]["is_infected"][i] == 0
        #    assert data["agent"]["infection_time"][i] == 0.0
        #    assert data["agent"]["symptoms"]["next_stage"][i] == 1.0
