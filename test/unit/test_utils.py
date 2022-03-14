from torch_june.utils import generate_erdos_renyi
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

