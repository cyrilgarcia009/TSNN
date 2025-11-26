import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        self.train_corr = []
        self.test_corr = []

    def train_loop(self, dataloader):
        self.model.train()
        num_batches = len(dataloader)
        train_loss = 0
        train_corr = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            # print(pred.shape, y.shape)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()
            train_corr += (np.corrcoef(pred.detach().flatten().to('cpu'),
                                       y.detach().flatten().to('cpu'))[0][1])

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss /= num_batches
        train_corr /= num_batches
        self.train_loss.append(train_loss)
        self.train_corr.append(train_corr)

    def test_loop(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        test_loss = 0
        test_corr = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                test_corr += (np.corrcoef(pred.flatten().to('cpu'), y.flatten().to('cpu'))[0][1])


        test_loss /= num_batches
        test_corr /= num_batches
        self.test_loss.append(test_loss)
        self.test_corr.append(test_corr)

    def fit(self, train, test=None, epochs=40, plot=True):
        for t in tqdm(range(epochs)):
            self.train_loop(train)
            if test is not None:
                self.test_loop(test)
        if plot:
            pd.concat([pd.Series(self.train_loss).rename('train_loss'),
                       pd.Series(self.test_loss).rename('test_loss')],
                      axis=1).plot()
            plt.title('MSE over Epochs')
            plt.show()
            pd.concat([pd.Series(self.train_corr).rename('train_loss'),
                       pd.Series(self.test_corr).rename('test_loss')],
                      axis=1).plot()
            plt.title('Correlation over Epochs')
            plt.show()

    def predict(self, dataloader):
        if isinstance(dataloader, DataLoader):
            non_shuffled = DataLoader(dataloader.dataset, batch_size=1024, num_workers=4, pin_memory=True,
                                      shuffle=False, collate_fn=dataloader.collate_fn)
        else:
            raise UserWarning('dataloader should be a torch Dataset')

        preds = []
        self.model.eval()

        with torch.inference_mode():
            for batch in non_shuffled:
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                X = X.to(self.device, non_blocking=True)

                pred = self.model(X)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        #return preds.flatten()

            # === NEW: Handle different output dimensions ===
        if preds.ndim == 2:
            return preds.flatten()

        elif preds.ndim == 3:
            preds_last_timestep = preds[:, -1, :]
            return preds_last_timestep.flatten()

        else:
            raise ValueError(
                f"Model output has unsupported number of dimensions: {preds.ndim}. "
                f"Got shape: {preds.shape}"
            )
            

    def score(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        return np.corrcoef(self.predict(dataloader), y)[0][1]
