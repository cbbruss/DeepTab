# Testing Model
import torch
from torch.utils.data import TensorDataset, DataLoader

from frustanet.model import FrustaNetRegression


def test_forward():
    net = FrustaNetRegression(n_features=10)
    x = torch.randn(1, 10)
    out = net(x)
    assert out.shape[1]==1

# TODO fix unit test

# def test_training_step():
#     net = FrustaNetRegression(n_features=10)
#     x = torch.randn(1, 10)
#     y = torch.randn(1, 1)
#     data =  torch.utils.data.TensorDataset(x, y)
#     data_iter = DataLoader(data)
#     loss = net.training_step(data, 0)
#     assert 3==5