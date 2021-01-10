# Testing Model
import torch

from frustanet.model import FrustaNetRegression


def test_forward():
    net = FrustaNetRegression(input_size=10)
    x = torch.randn(1, 10)
    out = net(x)
    assert out.shape[1]==1