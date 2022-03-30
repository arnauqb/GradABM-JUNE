import numpy as np
import torch

from torch_june.default_parameters import make_parameters, convert_lognormal_parameters


def test__lognormal_converter():
    mean = 4.5
    std = 1.5
    loc, scale = convert_lognormal_parameters(mean, std)
    dist = torch.distributions.LogNormal(loc, scale)
    assert np.isclose(dist.mean.item(), mean)
    assert np.isclose(dist.stddev.item(), std)


def test__parameters():
    params = make_parameters()
    sp = params["symptoms"]
    assert len(sp["stages"]) == 8
    for key in sp["transition_probabilities"]:
        assert key in sp["stages"]
        if key == "exposed":
            assert len(sp["transition_probabilities"][key]) == 1
        else:
            assert len(sp["transition_probabilities"][key]) == 10
    for key in sp["symptom_transition_times"]:
        assert key in sp["stages"]
