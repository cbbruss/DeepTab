import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule

class FrustaNetRegression(LightningModule):

    def __init__(self, n_features):
        super().__init__()

        """Let's start with one linear model and one
            non-linear model.

            Args:
                input_size: dimensions of input
        """
        self.linear = torch.nn.Linear(n_features, 1)

        # self.layer_2 = torch.nn.Linear(128, 256)
        # self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        """
            Take in a value of x and return regression
            value.

            Args:
                x: model inputs
            Returns:
                out: model prediction
        """
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):
        mseloss = nn.MSELoss()
        x, y = batch
        y_hat = self.forward(x)
        loss = mseloss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
