from torch_june.utils import generate_erdos_renyi, parse_age_probabilities
import torch

import numpy as np

class TestCustomErdosRenyi:
    def test__construction(self):
        vertices = torch.arange(500)
        edge_index = generate_erdos_renyi(vertices, 0.3)
        # Expected number of edges per node is 500 * 499 * 0.3
        expected = 0.3 * 499
        for v in vertices:
            v_edges = np.where(edge_index[0,:] == v)[0]
            assert np.isclose(len(v_edges), expected, rtol=0.3)
        expected_total = 500 * 499 * 0.3
        assert np.isclose(edge_index.shape[1], expected_total, rtol=0.1)

class TestParseProbabilities:
    def test__parsing(self):
        input = {"5-20" :0.2, "25-40": 0.7}
        probs = parse_age_probabilities(input)
        assert (probs[:5] == np.zeros(5)).all()
        assert (probs[5:20] == 0.2 * np.ones(15)).all()
        assert (probs[20:25] == np.zeros(5)).all()
        assert (probs[25:40] == 0.7 * np.ones(15)).all()
        assert (probs[40:] == np.zeros(60)).all()

