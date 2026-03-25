import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .. import utils


class TorchWrapper:
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss(), device='mps', grad_accum_steps=1):
        """
        Wrapper class around any pytorch custom model to standardize fit and predict steps
        :param model: pytorch model
        :param optimizer: pytorch optimizer
        :param loss_fn: loss function to sue with the model
        :param device: cpu, cuda or mps
        :param grad_accum_steps: for large models, enables to avoid memory errors by accumulating the gradients
        """
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        self.train_loss = []
        self.test_loss = []
        self.train_corr = []
        self.test_corr = []

    def _forward_model(self, X, pad_mask=None):
        if pad_mask is None:
            return self.model(X)
        return self.model(X, pad_mask=pad_mask)

    def _masked_loss(self, pred, y, pad_mask=None):
        if pad_mask is None or pred.ndim != 3 or y.ndim != 3:
            return self.loss_fn(pred, y)
        return self.loss_fn(pred[pad_mask], y[pad_mask])

    def _masked_corr(self, pred, y, pad_mask=None):
        if pad_mask is not None and pred.ndim == 3 and y.ndim == 3:
            pred = pred[pad_mask]
            y = y[pad_mask]
        return np.corrcoef(pred.detach().flatten().to('cpu'), y.detach().flatten().to('cpu'))[0][1]

    def train_loop(self, dataloader):
        self.model.train()
        num_batches = len(dataloader)
        train_loss = 0
        train_corr = 0
        accumulation_steps = 0

        for batch, data in enumerate(dataloader):
            if len(data) == 3:
                X, y, pad_mask = data
                pad_mask = pad_mask.to(self.device)
            else:
                X, y = data
                pad_mask = None
            X, y = X.to(self.device), y.to(self.device)

            pred = self._forward_model(X, pad_mask=pad_mask)
            loss = self._masked_loss(pred, y, pad_mask=pad_mask)
            loss = loss / self.grad_accum_steps

            train_loss += loss.item() * self.grad_accum_steps
            train_corr += self._masked_corr(pred, y, pad_mask=pad_mask)

            loss.backward()

            accumulation_steps += 1
            if accumulation_steps % self.grad_accum_steps == 0 or batch == num_batches - 1:
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
            for data in dataloader:
                if len(data) == 3:
                    X, y, pad_mask = data
                    pad_mask = pad_mask.to(self.device)
                else:
                    X, y = data
                    pad_mask = None
                X, y = X.to(self.device), y.to(self.device)
                pred = self._forward_model(X, pad_mask=pad_mask)
                test_loss += self._masked_loss(pred, y, pad_mask=pad_mask).item()
                test_corr += self._masked_corr(pred, y, pad_mask=pad_mask)

        test_loss /= num_batches
        test_corr /= num_batches
        self.test_loss.append(test_loss)
        self.test_corr.append(test_corr)

    def fit(self, train, test=None, epochs=40, plot=True, grad_accum_steps=None, verbose=1):
        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps

        if verbose >= 1:
            for t in tqdm(range(epochs)):
                self.train_loop(train)
                if test is not None:
                    self.test_loop(test)
        else:
            for t in range(epochs):
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
            non_shuffled = DataLoader(dataloader.dataset, batch_size=32, num_workers=4, pin_memory=True,
                                      shuffle=False, collate_fn=dataloader.collate_fn)
        else:
            raise UserWarning('dataloader should be a torch Dataset')

        preds = []
        self.model.eval()

        with torch.inference_mode():
            for batch in non_shuffled:
                if isinstance(batch, (list, tuple)):
                    X = batch[0]
                    pad_mask = batch[2] if len(batch) == 3 else None
                else:
                    X = batch
                    pad_mask = None
                X = X.to(self.device, non_blocking=True)
                if pad_mask is not None:
                    pad_mask = pad_mask.to(self.device, non_blocking=True)

                pred = self._forward_model(X, pad_mask=pad_mask)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        # return preds.flatten()

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


class MSELossWithL1Sparsity(nn.Module):
    def __init__(self, model, lambda_l1=0.01, reduction='mean'):
        super().__init__()
        self.model = model
        self.lambda_l1 = lambda_l1
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)

        gated_coeffs_list = self.model.get_gated_coeffs()

        l1_penalty = 0.0

        if len(gated_coeffs_list) > 0:
            for gated_coeffs in gated_coeffs_list:
                # l1_penalty += torch.sum(torch.abs(gated_coeffs))

                coef_mean = torch.mean(gated_coeffs)
                # l1_penalty -= ttorchorch.sum(torch.abs(gated_coeffs-coef_mean))
                # l1_penalty -= torch.sum(torch.clamp(gated_coeffs - coef_mean, 0, 1)**2)
                l1_penalty -= torch.sum(gated_coeffs ** 2)
        else:
            raise ValueError('No gated coefficients found')

        total_loss = mse + self.lambda_l1 * l1_penalty

        return total_loss
