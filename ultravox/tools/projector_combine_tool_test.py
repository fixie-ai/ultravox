import torch
import torch.nn as nn

from ultravox.tools import projector_combine_tool


def test_combine_linear_layers():
    linear_1 = nn.Linear(10, 12, bias=False)
    linear_2 = nn.Linear(12, 23, bias=False)
    combined = projector_combine_tool.combine_linear_layers(linear_1, linear_2)
    test_inp = torch.randn(1, combined.in_features)
    assert combined.in_features == linear_1.in_features
    assert combined.out_features == linear_2.out_features
    assert torch.allclose(linear_2(linear_1(test_inp)), combined(test_inp), atol=1e-4)
