import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset, Subset


class TorchDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class TorchDatasetRolling(Dataset):
    def __init__(self, X, y=None, n=10):
        self.X = X
        self.y = y
        self.n = n

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        start = max(0, idx - self.n + 1)
        if self.y is not None:
            return self.X[start:idx + 1], self.y[idx]
        else:
            return self.X[start:idx + 1]


def collate_pad_beginning(batch, pad_value=0.0, max_len=None):
    Xs, ys = zip(*batch)

    max_m = max_len or max(x.shape[0] for x in Xs)
    N, K = Xs[0].shape[1], Xs[0].shape[2]

    batch_size = len(Xs)
    X_padded = torch.full((batch_size, max_m, N, K), pad_value)
    mask = torch.zeros((batch_size, max_m), dtype=torch.bool)

    for i, x in enumerate(Xs):
        m = x.shape[0]
        X_padded[i, -m:] = x
        mask[i, -m:] = True

    y_batch = torch.stack(ys)
    return X_padded, y_batch#, mask


def np_to_torch(X, y=None, train_test_split=True, train_pct=0.7, batch_size=64, shuffle=True, n_rolling=1,
                ts_split=True):
    """
    converts tensors to a torch dataset with option of having a train/test split
    :param X:
    :param y:
    :param train_test_split:
    :param train_pct:
    :param batch_size:
    :return:
    """
    if n_rolling == 1:
        full_data = TorchDataset(X, y)
    else:
        full_data = TorchDatasetRolling(X, y, n=n_rolling)
    if train_test_split:
        train_size = int(train_pct * len(full_data))
        test_size = len(full_data) - train_size

        if ts_split:
            total_size = len(full_data)

            train_indices = list(range(train_size))
            test_indices = list(range(train_size, total_size))

            train_data = Subset(full_data, train_indices)
            test_data = Subset(full_data, test_indices)
        else:
            train_data, test_data = random_split(full_data, [train_size, test_size],
                                                 generator=torch.Generator().manual_seed(42))

        if n_rolling == 1:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
        else:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                      collate_fn=collate_pad_beginning
                                      )
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle,
                                     collate_fn=collate_pad_beginning
                                     )
        return train_loader, test_loader
    else:
        if n_rolling == 1:
            return DataLoader(full_data, batch_size=batch_size, shuffle=shuffle)
        else:
            return DataLoader(full_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pad_beginning)


def torch_to_np(d):
    """
    converts a torch dataset to a numpy matrix on which usual ML algo can be applied
    :param d: pytorch dataset
    :return: X, y as numpy arrays
    """
    indices = d.dataset.indices
    if hasattr(d.dataset.dataset, 'y'):
        X_np, y_np = d.dataset.dataset.X[indices], d.dataset.dataset.y[indices]
        y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1])
        X_np = torch.reshape(X_np, (X_np.shape[0] * X_np.shape[1], X_np.shape[2]))
        return X_np, y_np

    else:
        X_np = d.dataset.dataset.X[indices]
        X_np = torch.reshape(X_np, (X_np.shape[0] * X_np.shape[1], X_np.shape[2]))
        return X_np
