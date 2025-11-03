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
        self.y_linear = None
        self.y_conditional = None
        self.y_shift = None
        self.y_seasonal = None

    def generate_covar(self, n: int) -> torch.tensor:
        """

        :param n: covar dim
        :return:
        """
        A = np.random.normal(size=n ** 2).reshape(n, n)
        cov_raw = A @ A.T + np.eye(n) * 1e-6
        norm = np.diag(1 / np.sqrt(np.diag(cov_raw)))
        return torch.from_numpy(norm @ cov_raw @ norm)

    def generate_dataset(self, pct_zero_corr: float = 0.5, split_conditional: float = 0.3, split_shift: float = 0.0,
                         split_seasonal: float = 0.0) -> tuple:
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
        self.y_pred_optimal = y * 0
        self.y_linear = y * 0
        self.y_conditional = y * 0
        self.y_shift = y * 0
        self.y_seasonal = y * 0

        # pick features correl - linear relationship
        corr_with_y = np.random.uniform(low=0.005, high=0.03, size=self.n_f)
        corr_with_y *= np.random.choice([-1, 1], self.n_f)
        # zero out some of the features correl
        corr_with_y[int(self.n_f * pct_zero_corr):] = 0


        self.corr_with_y = torch.tensor(corr_with_y, dtype=torch.float32)

        X = np.array([np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                                    cov=self.generate_covar(self.n_ts),
                                                    size=self.T) for k in range(self.n_f)])

        start_linear_ft = 0
        n_corr_ft = int(self.n_f * (1 - pct_zero_corr))

        start_cond_ft = 0
        end_cond_ft = int(n_corr_ft * split_conditional)

        start_shift_ft = end_cond_ft
        end_shift_ft = int(split_shift * n_corr_ft) + start_shift_ft

        start_seasonal_ft = end_shift_ft
        end_seasonal_ft = int(split_seasonal * n_corr_ft) + start_seasonal_ft


        start_random_ft = int(self.n_f * (1-pct_zero_corr))

        # linear relationships
        for k in range(start_random_ft):
            X[k] = X[k] + corr_with_y[k] * y
            # self.y_pred_optimal += corr_with_y[k] * X[k]
            # self.y_linear += corr_with_y[k] * X[k]

        # Conditioning relationships (need to condition feature k by sign of feature)
        for k in range(start_cond_ft, end_cond_ft):
            X[k] = X[k] * np.sign(X[start_random_ft + k])
            optimal_pred = corr_with_y[k] * X[k] * np.sign(X[start_random_ft + k])
            self.y_pred_optimal += optimal_pred
            self.y_conditional += optimal_pred

        # Shift relationships
        for k in range(start_shift_ft, end_shift_ft):
            n_shift = 1
            X[k] = np.concatenate((X[k][n_shift:], X[k][:n_shift] * 0))
            optimal_pred = np.concatenate((X[k][:n_shift] * 0, X[k][n_shift:])) * corr_with_y[k]
            self.y_pred_optimal += optimal_pred
            self.y_shift += optimal_pred

        # Seasonal relationships
        for k in range(start_seasonal_ft, end_seasonal_ft):
            period = 10
            optimal_pred = X[k] * 0
            for i in range(len(X)):
                X[k][i] = X[k][i] - corr_with_y[k] * y[i] * (i % period != 0)
                optimal_pred[i] = corr_with_y[k] * X[k][i] * (i % period == 0)
            self.y_pred_optimal += optimal_pred
            self.y_seasonal += optimal_pred

        X = torch.from_numpy(X)
        X = torch.transpose(X, 0, 1)
        X = torch.transpose(X, 1, 2)
        X = X.to(dtype=torch.float32)

        self.y_pred_optimal = X @ self.corr_with_y
        return X, torch.from_numpy(y).to(dtype=torch.float32)
