import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class FrustaNetRegression(LightningModule):

  def __init__(self, input_size):
    super().__init__()

    """Let's start with one linear model and one
        non-linear model.

        Args:
            input_size: dimensions of input
    """
    self.linear = torch.nn.Linear(input_size, 1)

    # self.layer_2 = torch.nn.Linear(128, 256)
    # self.layer_3 = torch.nn.Linear(256, 10)

  def forward(self, x):
    """
        Take in a value of x and return regression
        value.
    """
    out = self.linear(x)
    return out