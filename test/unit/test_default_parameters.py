import numpy as np
import torch

from grad_june.default_parameters import make_parameters, convert_lognormal_parameters


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
    for key in sp["stage_transition_probabilities"]:
        assert key in sp["stages"]
        if key in ["recovered", "susceptible", "exposed"]:
            assert len(sp["stage_transition_probabilities"][key]) == 1
        else:
            assert len(sp["stage_transition_probabilities"][key]) == 10
    for key in sp["stage_transition_times"]:
        assert key in sp["stages"]
