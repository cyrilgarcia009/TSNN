import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from .. import utils


class TorchWrapper:
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss(), device='mps'):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss = []
        self.test_loss = []

    def train_loop(self, dataloader):
        self.model.train()
        num_batches = len(dataloader)
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss /= num_batches
        self.train_loss.append(train_loss)

    def test_loop(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        self.test_loss.append(test_loss)

    def fit(self, train, test=None, epochs=40):
        for t in range(epochs):
            self.train_loop(train)
            if test is not None:
                self.test_loop(test)

    def predict(self, dataloader):
        if isinstance(dataloader, DataLoader):
            non_shuffled = DataLoader(dataloader.dataset, shuffle=False)
        else:
            raise UserWarning('dataloader should be a torch Dataset')

        res = []
        self.model.eval()
        with torch.no_grad():
            for z in non_shuffled:
                if len(z) == 2:
                    X = z[0]
                else:
                    X = z
                X = X.to(self.device)
                pred = self.model(X).cpu().numpy()
                res.append(pred.flatten())
        return np.concatenate(res)

    def score(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        return np.corrcoef(self.predict(dataloader), y)[0][1]
