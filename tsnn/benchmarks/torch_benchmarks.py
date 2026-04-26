import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
from .. import utils


class TorchWrapper:
    def __init__(self, model, optimizer, loss_fn=nn.MSELoss(), device=None, grad_accum_steps=1, scheduler=None):
        """
        Wrapper class around any pytorch custom model to standardize fit and predict steps
        :param model: pytorch model
        :param optimizer: pytorch optimizer
        :param loss_fn: loss function to sue with the model
        :param device: cpu, cuda or mps; auto-detected if None
        :param grad_accum_steps: for large models, enables to avoid memory errors by accumulating the gradients
        :param scheduler: optional pytorch scheduler stepped once per epoch after train/test loops
        """
        if device is None:
            device = (
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
            )
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.grad_accum_steps = grad_accum_steps
        self.scheduler = scheduler
        self.train_loss = []
        self.test_loss = []
        self.train_corr = []
        self.test_corr = []
        self.val_loss = []
        self.val_corr = []
        self.best_val_metric = None
        self.best_epoch = None

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

    def eval_loop(self, dataloader):
        self.model.eval()
        num_batches = len(dataloader)
        eval_loss = 0
        eval_corr = 0

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
                eval_loss += self._masked_loss(pred, y, pad_mask=pad_mask).item()
                eval_corr += self._masked_corr(pred, y, pad_mask=pad_mask)

        eval_loss /= num_batches
        eval_corr /= num_batches
        return eval_loss, eval_corr

    def _split_train_val_loader(self, train_loader, val_pct):
        if val_pct is None or val_pct <= 0:
            return train_loader, None
        if val_pct >= 1:
            raise ValueError("val_pct must be in [0, 1).")
        if not hasattr(train_loader, 'dataset') or not hasattr(train_loader.dataset, 'indices'):
            raise ValueError("Validation split currently requires a DataLoader backed by a torch.utils.data.Subset.")

        base_subset = train_loader.dataset
        dataset = base_subset.dataset
        indices = list(base_subset.indices)
        n_val = max(1, int(len(indices) * val_pct))
        if n_val >= len(indices):
            raise ValueError("val_pct leaves no samples for training.")

        train_indices = indices[:-n_val]
        val_indices = indices[-n_val:]

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        loader_kwargs = dict(
            batch_size=train_loader.batch_size,
            collate_fn=train_loader.collate_fn,
        )
        train_split_loader = DataLoader(train_subset, shuffle=False, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
        return train_split_loader, val_loader

    def _is_better_val_metric(self, metric_name, current, best):
        if best is None:
            return True
        if metric_name == 'loss':
            return current < best
        if metric_name == 'corr':
            return current > best
        raise ValueError("early_stopping_metric must be 'loss' or 'corr'.")

    def fit(
            self,
            train,
            test=None,
            epochs=40,
            plot=True,
            grad_accum_steps=None,
            verbose=1,
            scheduler=None,
            val_pct=0.0,
            early_stopping_patience=None,
            early_stopping_metric='loss',
            val_warmup_epochs=5,
    ):
        if grad_accum_steps is not None:
            self.grad_accum_steps = grad_accum_steps
        if scheduler is not None:
            self.scheduler = scheduler
        train_loader, val_loader = self._split_train_val_loader(train, val_pct)

        best_state = None
        best_metric = None
        best_epoch = None
        patience_counter = 0

        if verbose >= 1:
            progress = tqdm(range(epochs))
            for t in progress:
                self.train_loop(train_loader)
                if test is not None:
                    self.test_loop(test)
                if val_loader is not None:
                    val_loss, val_corr = self.eval_loop(val_loader)
                    self.val_loss.append(val_loss)
                    self.val_corr.append(val_corr)
                    if t >= val_warmup_epochs:
                        current_metric = val_loss if early_stopping_metric == 'loss' else val_corr
                        if self._is_better_val_metric(early_stopping_metric, current_metric, best_metric):
                            best_metric = current_metric
                            best_epoch = t
                            best_state = copy.deepcopy(self.model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                if self.scheduler is not None:
                    self.scheduler.step()
                postfix = {
                    'train_corr': f'{self.train_corr[-1]:.4f}',
                    'train_loss': f'{self.train_loss[-1]:.4f}',
                }
                if test is not None and self.test_corr:
                    postfix['test_corr'] = f'{self.test_corr[-1]:.4f}'
                if test is not None and self.test_loss:
                    postfix['test_loss'] = f'{self.test_loss[-1]:.4f}'
                if val_loader is not None and self.val_corr:
                    postfix['val_corr'] = f'{self.val_corr[-1]:.4f}'
                    postfix['val_loss'] = f'{self.val_loss[-1]:.4f}'
                if self.optimizer.param_groups:
                    postfix['lr'] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                progress.set_postfix(postfix)
                if (
                        val_loader is not None and
                        early_stopping_patience is not None and
                        t >= val_warmup_epochs and
                        patience_counter >= early_stopping_patience
                ):
                    progress.write(f"Early stopping at epoch {t + 1}; best epoch was {best_epoch + 1}.")
                    break
        else:
            for t in range(epochs):
                self.train_loop(train_loader)
                if test is not None:
                    self.test_loop(test)
                if val_loader is not None:
                    val_loss, val_corr = self.eval_loop(val_loader)
                    self.val_loss.append(val_loss)
                    self.val_corr.append(val_corr)
                    if t >= val_warmup_epochs:
                        current_metric = val_loss if early_stopping_metric == 'loss' else val_corr
                        if self._is_better_val_metric(early_stopping_metric, current_metric, best_metric):
                            best_metric = current_metric
                            best_epoch = t
                            best_state = copy.deepcopy(self.model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                if self.scheduler is not None:
                    self.scheduler.step()
                if (
                        val_loader is not None and
                        early_stopping_patience is not None and
                        t >= val_warmup_epochs and
                        patience_counter >= early_stopping_patience
                ):
                    break

        self.best_val_metric = best_metric
        self.best_epoch = best_epoch
        if best_state is not None:
            self.model.load_state_dict(best_state)

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
            if self.val_loss:
                pd.concat([
                    pd.Series(self.train_loss).rename('train_loss'),
                    pd.Series(self.val_loss).rename('val_loss'),
                    pd.Series(self.test_loss).rename('test_loss'),
                ], axis=1).plot()
                plt.title('MSE over Epochs (with validation)')
                plt.show()
                pd.concat([
                    pd.Series(self.train_corr).rename('train_corr'),
                    pd.Series(self.val_corr).rename('val_corr'),
                    pd.Series(self.test_corr).rename('test_corr'),
                ], axis=1).plot()
                plt.title('Correlation over Epochs (with validation)')
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


class PearsonCorrLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        pred = pred - pred.mean()
        target = target - target.mean()

        pred_std = torch.sqrt(torch.mean(pred ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target ** 2) + self.eps)

        corr = torch.mean((pred / pred_std) * (target / target_std))
        return 1.0 - corr


class MSECorrLoss(nn.Module):
    def __init__(self, alpha=0.1, eps=1e-8, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, pred, target):
        mse = self.mse(pred, target)

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        pred = pred - pred.mean()
        target = target - target.mean()

        pred_std = torch.sqrt(torch.mean(pred ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target ** 2) + self.eps)

        corr = torch.mean((pred / pred_std) * (target / target_std))
        return mse + self.alpha * (1.0 - corr)
