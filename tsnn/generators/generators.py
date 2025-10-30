import numpy as np
import torch


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

    def generate_dataset(self, pct_zero_corr: float = 0.5):
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
        self.corr_with_y = torch.tensor(corr_with_y, dtype=torch.float32)

        X = np.array([np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                                    cov=self.generate_covar(self.n_ts),
                                                    size=self.T) + corr_with_y[k] * y for k in range(self.n_f)])
        X = torch.from_numpy(X)
        X = torch.transpose(X, 0, 1)
        X = torch.transpose(X, 1, 2)
        X = X.to(dtype=torch.float32)

        self.y_pred_optimal = X @ self.corr_with_y
        return X, torch.from_numpy(y).to(dtype=torch.float32)
