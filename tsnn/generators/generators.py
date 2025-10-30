import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Generator:
    def __init__(self, T: int, n_ts: int, n_f: int):
        """
        generates a dataset (X, y) where X is a tensor of dim (T, n_ts, n_f) and y (T, n_ts)
        :param T: time dimension
        :param n_ts: number of time-series
        :param n_f: number of features
        """
        self.n_ts = n_ts
        self.n_f = n_f
        self.T = T
        self.corr_with_y = None
        self.y_pred_optimal = None

    def generate_covar(self, n: int) -> torch.tensor:
        """

        :param n: covar dim
        :return:
        """
        A = np.random.normal(size=n ** 2).reshape(n, n)
        cov_raw = A @ A.T + np.eye(n) * 1e-6
        norm = np.diag(1 / np.sqrt(np.diag(cov_raw)))
        return torch.from_numpy(norm @ cov_raw @ norm)

    def generate_dataset(self, pct_zero_corr:float = 0.5):
        """

        :param pct_zero_corr: percentage of features with 0 correlation to y
        :return: torch datasets
        """
        # todo: add some conditioning predictive power (only direct correlation of features for now)
        # todo: add non-linear dependencies
        # todo: add lead-lag alpha between time series
        # random y
        y = np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                          cov=self.generate_covar(self.n_ts),
                                          size=self.T)
        # pick features correl
        corr_with_y = np.random.uniform(low=0.005, high=0.03, size=self.n_f)
        corr_with_y *= np.random.choice([-1, 1], self.n_f)
        # zero out some of the features correl
        corr_with_y[int(self.n_f * pct_zero_corr):] = 0
        self.corr_with_y = torch.tensor(corr_with_y)

        X = np.array([np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                                    cov=self.generate_covar(self.n_ts),
                                                    size=self.T) + corr_with_y[k] * y for k in range(self.n_f)])
        X = torch.from_numpy(X)
        X = torch.transpose(X, 0, 1)
        X = torch.transpose(X, 1, 2)

        self.y_pred_optimal = X @ self.corr_with_y
        return X, torch.from_numpy(y)

    def np_to_torch(self, X, y, train_test_split=True, train_pct=0.7, batch_size=64):
        """
        converts tensors to a torch dataset with option of having a train/test split
        :param X:
        :param y:
        :param train_test_split:
        :param train_pct:
        :param batch_size:
        :return:
        """
        full_data = TorchDataset(X, y)
        if train_test_split:
            train_size = int(train_pct * len(full_data))
            test_size = len(full_data) - train_size
            train_data, test_data = random_split(full_data, [train_size, test_size],
                                                 generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size)
            return train_loader, test_loader
        else:
            return DataLoader(full_data, batch_size=batch_size, shuffle=True)

    def torch_to_np(sel, d):
        """
        converts a torch dataset to a numpy matrix on which usual ML algo can be applied
        :param d: pytorch dataset
        :return: X, y as numpy arrays
        """
        X_np, y_np = d.dataset.dataset.X, d.dataset.dataset.y
        X_np = torch.reshape(X_np, (X_np.shape[0] * X_np.shape[1], X_np.shape[2]))
        y_np = y_np.reshape(y_np.shape[0] * y_np.shape[1])
        return X_np, y_np
