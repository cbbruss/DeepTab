# Testing Model
import torch

from frustanet.model import FrustaNetRegression


def test_forward():
    net = FrustaNetRegression(n_features=10)
    x = torch.randn(2, 10)
    out = net(x)
    assert out[0].shape[1] == 1

def test_training_step():
    net = FrustaNetRegression(n_features=10)
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    loss = net.training_step((x, y), 0)
    assert loss > 0