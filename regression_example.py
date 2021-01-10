import os
import torch
from torch import nn, tensor
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


from frustanet.model import FrustaNetRegression


def main():

    X, y = load_boston(return_X_y=True)
    n_features = X.shape[1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = tensor(X, dtype=torch.float)
    y_train = tensor(y, dtype=torch.float)


    datasets = TensorDataset(X_train, y_train)

    train_iter = DataLoader(datasets, batch_size=10, shuffle=True)
    f_model = FrustaNetRegression(n_features=n_features)
    trainer = Trainer()
    trainer.fit(f_model, train_iter)

if __name__ == '__main__':
    main()