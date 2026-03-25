from typing import Optional

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np


class TorchDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, add_noise: bool = False,
                 noise_scale: float = 0.5):
        self.X = X
        self.y = y
        self.add_noise = add_noise
        self.noise_scale = noise_scale

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            if self.add_noise:
                return self.X[idx] + self.noise_scale * np.random.normal(size=self.X.shape[1]), self.y[idx]
            else:
                return self.X[idx], self.y[idx]
        else:
            if self.add_noise:
                return self.X[idx] + self.noise_scale * np.random.normal(size=self.X.shape[1])
            else:
                return self.X[idx]


class TorchDatasetRolling(Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None, n: int = 10, roll_y: bool = False,
                 add_noise: bool = False, noise_scale: float = 0.5):
        self.X = X
        self.y = y
        self.n = n
        self.roll_y = roll_y
        self.add_noise = add_noise
        self.noise_scale = noise_scale

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Build a left-aligned rolling window ending at idx.
        start = max(0, idx - self.n + 1)
        if self.y is not None:
            if self.roll_y:
                if self.add_noise:
                    noise_shape = self.X[start:idx + 1].shape
                    return (
                        self.X[start:idx + 1] + self.noise_scale * np.random.normal(
                            size=np.prod(noise_shape)).reshape(
                            noise_shape),
                        self.y[start:idx + 1])
                else:
                    return self.X[start:idx + 1], self.y[start:idx + 1]
            else:
                if self.add_noise:
                    noise_shape = self.X[start:idx + 1].shape
                    return (
                        self.X[start:idx + 1] + self.noise_scale * np.random.normal(
                            size=np.prod(noise_shape)).reshape(
                            noise_shape),
                        self.y[idx]
                    )
                else:
                    return self.X[start:idx + 1], self.y[idx]
        else:
            return self.X[start:idx + 1]


def collate_pad_beginning(batch, pad_value: float = 0.0, max_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Xs, ys = zip(*batch)
    batch_size = len(Xs)
    max_m = max_len or max(x.shape[0] for x in Xs)

    # Pad on the left so the most recent time steps stay aligned at the end.
    if len(Xs[0].shape) == 3:
        N, K = Xs[0].shape[1], Xs[0].shape[2]
        X_padded = torch.full((batch_size, max_m, N, K), pad_value)
    elif len(Xs[0].shape) == 2:
        N = Xs[0].shape[1]
        X_padded = torch.full((batch_size, max_m, N), pad_value)
    else:
        raise UserWarning('X tensor dont have the correct shape')
    mask = torch.zeros((batch_size, max_m), dtype=torch.bool)

    for i, x in enumerate(Xs):
        m = x.shape[0]
        X_padded[i, -m:] = x
        mask[i, -m:] = True

    max_m = max_len or max(x.shape[0] for x in ys)
    if len(ys[0].shape) == 2:
        N = ys[0].shape[1]
        y_padded = torch.full((batch_size, max_m, N), pad_value)
        mask = torch.zeros((batch_size, max_m), dtype=torch.bool)
        for i, x in enumerate(ys):
            m = x.shape[0]
            y_padded[i, -m:] = x
            mask[i, -m:] = True
    else:
        y_padded = torch.stack(ys)

    return X_padded, y_padded, mask


def np_to_torch(X, y=None, train_test_split=True, train_pct=0.625, batch_size=256, shuffle=True, n_rolling=1,
                ts_split=True, narrow=False, roll_y=False, add_noise=False, noise_scale=0.5):
    """
    converts tensors to a torch dataset with option of having a train/test split
    :param X:
    :param y:
    :param train_test_split:
    :param train_pct:
    :param batch_size:
    :return:
    """
    if narrow and n_rolling == 1:
        dataset = TorchDataset(X.reshape((X.shape[0] * X.shape[1], X.shape[2])),
                               y.reshape((y.shape[0] * y.shape[1], 1)), add_noise=add_noise, noise_scale=noise_scale)
    elif narrow and (n_rolling > 1):
        dataset = TorchDatasetRolling(X.transpose(0, 1).reshape((X.shape[0] * X.shape[1], X.shape[2])),
                                      y.transpose(0, 1).reshape((y.shape[0] * y.shape[1], 1)),
                                      n=n_rolling, roll_y=roll_y, add_noise=add_noise, noise_scale=noise_scale)
    else:
        dataset = TorchDataset(X, y, add_noise=add_noise,
                               noise_scale=noise_scale) if n_rolling == 1 else TorchDatasetRolling(X, y, n=n_rolling,
                                                                                                   roll_y=roll_y,
                                                                                                   add_noise=add_noise,
                                                                                                   noise_scale=noise_scale)

    if not train_test_split:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_pad_beginning if n_rolling > 1 else None)

    train_size = int(train_pct * len(dataset))
    test_size = len(dataset) - train_size

    if ts_split:
        train_indices = range(train_size)
        test_indices = range(train_size, len(dataset))
        train_data, test_data = Subset(dataset, train_indices), Subset(dataset, test_indices)
    else:
        train_data, test_data = random_split(dataset, [train_size, test_size],
                                             generator=torch.Generator().manual_seed(42))

    collate_fn = collate_pad_beginning if n_rolling > 1 else None
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return train_loader, test_loader


def torch_to_np(d):
    """
    converts a torch dataset to a numpy matrix on which usual ML algo can be applied
    :param d: pytorch dataset
    :return: X, y as numpy arrays
    """
    indices = d.dataset.indices
    if hasattr(d.dataset.dataset, 'y'):
        X_np, y_np = d.dataset.dataset.X[indices], d.dataset.dataset.y[indices]
        if len(X_np.shape) == 3:
            y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1])
            X_np = torch.reshape(X_np, (X_np.shape[0] * X_np.shape[1], X_np.shape[2]))
        else:
            y_np = y_np.flatten()
        return X_np, y_np

    else:
        X_np = d.dataset.dataset.X[indices]
        X_np = torch.reshape(X_np, (X_np.shape[0] * X_np.shape[1], X_np.shape[2]))
        return X_np


def shift_torch(x: torch.Tensor, n: int) -> torch.Tensor:
    T = x.size(0)
    if n == 0:
        return x.clone()
    out = torch.zeros_like(x)

    if n > 0:
        out[n:] = x[:-n]
    else:
        k = -n
        out[:T - k] = x[k:]
    return out


def torch_to_np_features(d, n_rolling: int = 10):
    indices = d.dataset.indices
    dataset = d.dataset.dataset

    X = dataset.X[indices]
    assert X.ndim == 3  # (T, N, F)
    T, N, F = X.shape

    if hasattr(dataset, 'y'):
        y_np = dataset.y[indices]
        y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1])

    X_out = []

    for i in range(N):
        # Reorder stocks cyclically so each target series is seen first.
        X_reordered = torch.cat(
            [X[:, i:], X[:, :i]],
            dim=1
        )
        X_flat = X_reordered.reshape(T, -1)

        # Concatenate lagged views of the reordered features.
        X_lagged = torch.cat(
            [shift_torch(X_flat, n=k) for k in range(n_rolling)],
            dim=1
        )

        X_out.append(X_lagged)

    X_np = torch.cat(X_out, dim=1)
    X_np = X_np.reshape(T * N, -1)

    if hasattr(dataset, 'y'):
        return X_np, y_np

    return X_np


def generate_derangement(n: int) -> np.ndarray:
    rng = np.random.default_rng()
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
