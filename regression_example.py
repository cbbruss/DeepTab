import os
import torch
from torch import nn, tensor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


from frustanet.model import FrustaNetRegression


def main():

    X, y = fetch_california_housing(return_X_y=True)
    n_features = X.shape[1]

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid, y_test_valid, test_size=0.5, random_state=42)

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_train = (X_train - train_mean) / train_std
    X_valid = (X_valid - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    



    X_train = tensor(X_train, dtype=torch.float)
    X_valid = tensor(X_valid, dtype=torch.float)
    X_test = tensor(X_test, dtype=torch.float)
    y_train = tensor(y_train, dtype=torch.float).reshape(-1, 1)
    y_valid = tensor(y_valid, dtype=torch.float).reshape(-1, 1)
    y_test = tensor(y_test, dtype=torch.float).reshape(-1, 1)

    train_datasets = TensorDataset(X_train, y_train)
    valid_datasets = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(train_datasets, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_datasets)
    f_model = FrustaNetRegression(n_features=n_features)
    trainer = Trainer(callbacks=[EarlyStopping(monitor='Val Loss')])
    trainer.fit(f_model, train_loader, valid_loader)
    f_preds = f_model.forward(X_valid)
    print(np.mean((f_preds[0].flatten().detach().numpy() - y_valid.flatten().numpy()) ** 2))

    ## Training Comparison GBM
    reg = GradientBoostingRegressor(random_state=0, verbose=True)

    reg.fit(X_train, y_train)

    preds = reg.predict(X_valid)

    print(np.mean((preds - y_valid.flatten().numpy()) ** 2))

if __name__ == '__main__':
    main()