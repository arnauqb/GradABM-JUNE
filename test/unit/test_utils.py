from torch_june.utils import parse_age_probabilities
import numpy as np


class TestParseProbabilities:
    def test__parsing(self):
        input = {"5-20": 0.2, "25-40": 0.7}
        probs = parse_age_probabilities(input)
        assert (probs[:5] == np.zeros(5)).all()
        assert (probs[5:20] == 0.2 * np.ones(15)).all()
        assert (probs[20:25] == np.zeros(5)).all()
        assert (probs[25:40] == 0.7 * np.ones(15)).all()
        assert (probs[40:] == np.zeros(60)).all()
