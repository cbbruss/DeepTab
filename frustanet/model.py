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

        self.f2_layer0 = torch.nn.Linear(n_features, n_features*2)
        self.f2_bnorm = torch.nn.BatchNorm1d(n_features*2)
        self.f2_layer1 = torch.nn.Linear(n_features*2, 1)

        self.f3_layer0 = torch.nn.Linear(n_features, n_features*4)
        self.f3_bnorm0 = torch.nn.BatchNorm1d(n_features*4)
        self.f3_layer1 = torch.nn.Linear(n_features*4, n_features*2)
        self.f3_bnorm1 = torch.nn.BatchNorm1d(n_features*2)
        self.f3_layer2 = torch.nn.Linear(n_features*2, 1)

        self.f4_layer0 = torch.nn.Linear(n_features, n_features*8)
        self.f4_bnorm0 = torch.nn.BatchNorm1d(n_features*8)
        self.f4_layer1 = torch.nn.Linear(n_features*8, n_features*4)
        self.f4_bnorm1 = torch.nn.BatchNorm1d(n_features*4)
        self.f4_layer2 = torch.nn.Linear(n_features*4, n_features*2)
        self.f4_bnorm2 = torch.nn.BatchNorm1d(n_features*2)
        self.f4_layer3 = torch.nn.Linear(n_features*2, 1)


    def forward(self, x):
        """
            Take in a value of x and return regression
            value.

            Args:
                x: model inputs
            Returns:
                out: model prediction
        """

        out_linear = self.linear(x)

        out_f2 = self.f2_layer0(x)
        out_f2 = self.f2_bnorm(out_f2)
        out_f2 = torch.tanh(out_f2)
        out_f2 = self.f2_layer1(out_f2)

        out_f3 = self.f3_layer0(x)
        out_f3 = self.f3_bnorm0(out_f3)
        out_f3 = torch.tanh(out_f3)
        out_f3 = self.f3_layer1(out_f3)
        out_f3 = self.f3_bnorm1(out_f3)
        out_f3 = torch.tanh(out_f3)
        out_f3 = self.f3_layer2(out_f3)

        out_f4 = self.f4_layer0(x)
        out_f4 = self.f4_bnorm0(out_f4)
        out_f4 = torch.tanh(out_f4)
        out_f4 = self.f4_layer1(out_f4)
        out_f4 = self.f4_bnorm1(out_f4)
        out_f4 = torch.tanh(out_f4)
        out_f4 = self.f4_layer2(out_f4)
        out_f4 = self.f4_bnorm2(out_f4)
        out_f4 = torch.tanh(out_f4)
        out_f4 = self.f4_layer3(out_f4)

        out_final = out_linear + out_f2 + out_f3 + out_f4
        # out_final = out_linear


        return out_final, out_linear, out_f2, out_f3, out_f4

    def training_step(self, batch, batch_idx):
        mseloss = nn.MSELoss()
        x, y = batch
        y_hat, y_linear, y_f2, y_f3, y_f4 = self(x)
        l_linear = mseloss(y_linear, y)
        l_f2 = mseloss(y_f2, y)
        l_f3 = mseloss(y_f3, y)
        l_f4 = mseloss(y_f4, y)
        loss = mseloss(y_hat, y)

        self.log('Loss', loss, on_step=True)
        self.log('Linear Loss', l_linear, on_step=True)
        self.log('F2 Loss', l_f2, on_step=True)
        self.log('F3 Loss', l_f3, on_step=True)
        self.log('F4 Loss', l_f4, on_step=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001, weight_decay=0.1)

    def validation_step(self, batch, batch_idx):
        mseloss = nn.MSELoss()
        x, y = batch
        y_hat, y_linear, y_f2, y_f3, y_f4 = self(x)
        loss = mseloss(y_hat, y)
        self.log('Val Loss', loss)

