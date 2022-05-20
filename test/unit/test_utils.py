import pytest
import numpy as np
import datetime
import torch
import random

from torch_june.utils import parse_age_probabilities, read_date, fix_seed


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
        a1 = torch.rand(10,)
        b1 = np.random.rand(10,)
        c1 = random.random()
        fix_seed(1992)
        a2 = torch.rand(10,)
        b2 = np.random.rand(10,)
        c2 = random.random()
        assert (a1 == a2).all()
        assert (b1 == b2).all()
        assert c1 == c2
