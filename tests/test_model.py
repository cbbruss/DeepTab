# Testing Model
import torch
import numpy as np

from frustanet.model import FrustaNetRegression


# Tests with just linear model
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

def test_validation_step():
    net = FrustaNetRegression(n_features=10)
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    loss = net.validation_step((x, y), 0)
    print(loss)
    assert loss > 0

# Tests with non-linear model
def test_forward_nonlinear():
    net = FrustaNetRegression(n_features=10, n_estimators=5)
    x = torch.randn(2, 10)
    out = net(x)
    assert out[0].shape[1] == 1

def test_training_step_nonlinear():
    net = FrustaNetRegression(n_features=10, n_estimators=5)
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    loss = net.training_step((x, y), 0)
    assert loss > 0

def test_validation_step_nonlinear():
    net = FrustaNetRegression(n_features=10, n_estimators=5)
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    loss = net.validation_step((x, y), 0)
    assert loss > 0

def test_predict():
    net = FrustaNetRegression(n_features=10, n_estimators=5)
    x = torch.randn(2, 10)
    y = torch.randn(2, 1)
    preds = net.predict(x)
    mse = torch.mean((preds - y) ** 2)
    assert len(preds) == 2
    assert mse