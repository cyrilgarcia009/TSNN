import numpy as np
import torch
from .. import utils
import pandas as pd


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
        self.X = None
        self.y = None
        self.corr_with_y = None
        self.ys = {'true': torch.empty(0), 'optimal': torch.empty(0), 'linear': torch.empty(0),
                   'conditional': torch.empty(0), 'shift': torch.empty(0), 'seasonal': torch.empty(0),
                   'cs': torch.empty(0), 'cs_shift': torch.empty(0)}
        self.y_pred_optimal = None
        self.y_linear = None
        self.y_conditional = None
        self.y_shift = None
        self.y_seasonal = None
        self.y_cs = None
        self.y_cs_shift = None
        self.dataloader = None
        self.is_transposed = False

    def generate_covar(self, n: int) -> torch.Tensor:
        """

        :param n: covar dim
        :return:
        """
        A = np.random.normal(size=n ** 2).reshape(n, n)
        cov_raw = A @ A.T * 0 + np.eye(n) * 1e-2
        norm = np.diag(1 / np.sqrt(np.diag(cov_raw)))
        return torch.from_numpy(norm @ cov_raw @ norm)

    def generate_dataset(self, pct_zero_corr: float = 0.5, split_conditional: float = 0.3, split_shift: float = 0.0,
                         split_seasonal: float = 0.0, split_cs: float = 0.0, split_cs_shift: float = 0.0,
                         low_corr: float = 0.005, high_corr: float = 0.03) -> None:
        """

        :param pct_zero_corr: percentage of features with 0 correlation to y
        :return: torch datasets
        """
        # todo: add non-linear dependencies
        assert split_conditional + split_shift + split_seasonal + split_cs + split_cs_shift <= 1
        y = np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                          cov=self.generate_covar(self.n_ts),
                                          size=self.T)
        self.y_pred_optimal = y * 0
        self.y_linear = y * 0
        self.y_conditional = y * 0
        self.y_shift = y * 0
        self.y_seasonal = y * 0
        self.y_cs = y * 0
        self.y_cs_shift = y * 0

        # pick features correl - linear relationship
        corr_with_y = np.random.uniform(low=low_corr, high=high_corr, size=self.n_f)
        corr_with_y *= np.random.choice([-1, 1], self.n_f)
        # zero out some of the features correl
        corr_with_y[int(self.n_f * (1-pct_zero_corr)):] = 0

        self.corr_with_y = torch.tensor(corr_with_y, dtype=torch.float32)

        # X is an array containing the features, each feature is a matrix of dimension (T, n_ts)
        X = np.array([np.random.multivariate_normal(mean=np.zeros(self.n_ts),
                                                    cov=self.generate_covar(self.n_ts),
                                                    size=self.T) for k in range(self.n_f)])

        start_linear_ft = 0
        n_corr_ft = int(self.n_f * (1 - pct_zero_corr))

        start_cond_ft = 0
        end_cond_ft = int(split_conditional * n_corr_ft)

        start_shift_ft = end_cond_ft
        end_shift_ft = int(split_shift * n_corr_ft) + start_shift_ft

        start_seasonal_ft = end_shift_ft
        end_seasonal_ft = int(split_seasonal * n_corr_ft) + start_seasonal_ft

        start_cs_ft = end_seasonal_ft
        end_cs_ft = int(split_cs * n_corr_ft) + start_cs_ft

        start_cs_shift_ft = end_cs_ft
        end_cs_shift_ft = int(split_cs_shift * n_corr_ft) + start_cs_shift_ft

        start_random_ft = int(self.n_f * (1 - pct_zero_corr))

        # linear relationships
        for k in range(start_random_ft):
            X[k] += corr_with_y[k] * y

        # CS relationships
        for k in range(start_cs_ft, end_cs_ft):
            X[k] = X[k] - corr_with_y[k] * y
            y_cs = np.concatenate((y[:, 1:], y[:, :1]), axis=1)
            # stock n feature k predicts stock n+1
            X[k] = X[k] + corr_with_y[k] * y_cs

        # Conditioning relationships (need to condition feature k by sign of feature)
        for k in range(start_cond_ft, end_cond_ft):
            X[k] = X[k] * np.sign(X[start_random_ft + k])
            optimal_pred = corr_with_y[k] * X[k] * np.sign(X[start_random_ft + k])
            self.y_pred_optimal += optimal_pred
            self.y_conditional += optimal_pred

        # Shift relationships - shift could be variable by feature
        for k in range(start_shift_ft, end_shift_ft):
            n_shift = 1
            y_shifted = np.concatenate((y[n_shift:], y[:n_shift] * 0))

            X[k] = X[k] - corr_with_y[k] * y
            X[k] = X[k] + corr_with_y[k] * y_shifted
            optimal_pred = np.concatenate((X[k][:n_shift] * 0, X[k][:-n_shift])) * corr_with_y[k]

            self.y_pred_optimal += optimal_pred
            self.y_shift += optimal_pred

        # Seasonal relationships - could also make the seasonal pattern random by feature
        for k in range(start_seasonal_ft, end_seasonal_ft):
            period = 10
            optimal_pred = X[k] * 0
            for i in range(X[k].shape[0]):
                X[k][i] = X[k][i] - corr_with_y[k] * y[i] * (i % period != 0)
                optimal_pred[i] = corr_with_y[k] * X[k][i] * (i % period == 0)
            self.y_pred_optimal += optimal_pred
            self.y_seasonal += optimal_pred

        # CS relationships
        for k in range(start_cs_ft, end_cs_ft):
            X[k] = X[k] - corr_with_y[k] * y
            cs_shift = np.concatenate((X[k][:, [-1]], X[k][:, 0:-1]), axis=1)
            self.y_pred_optimal += corr_with_y[k] * cs_shift
            self.y_cs += corr_with_y[k] * cs_shift

        # CS + Shift relationships - shift could be variable
        for k in range(start_cs_shift_ft, end_cs_shift_ft):
            X[k] = X[k] - corr_with_y[k] * y
            n_shift = 1

            y_shifted = np.concatenate((y[n_shift:], y[:n_shift] * 0))
            y_cs = np.concatenate((y[:, 1:], y[:, :1]), axis=1)
            y_shifted_cs = np.concatenate((y_shifted[:, 1:], y_shifted[:, :1]), axis=1)

            X[k] = X[k] - corr_with_y[k] * y_cs
            X[k] = X[k] + corr_with_y[k] * y_shifted_cs

            optimal_pred = np.concatenate((X[k][:n_shift] * 0, X[k][:-n_shift]))
            optimal_pred = np.concatenate((optimal_pred[:, [-1]], optimal_pred[:, :-1]), axis=1)
            optimal_pred = optimal_pred * corr_with_y[k]

            self.y_pred_optimal += optimal_pred
            self.y_cs_shift += optimal_pred

        # Adding the usual linear relationships to the optimal pred
        for k in range(end_cs_shift_ft, start_random_ft):
            self.y_pred_optimal += corr_with_y[k] * X[k]
            self.y_linear += corr_with_y[k] * X[k]

        X = torch.from_numpy(X)
        X = torch.transpose(X, 0, 1)
        X = torch.transpose(X, 1, 2)

        self.X = X.to(dtype=torch.float32)
        self.y = torch.from_numpy(y).to(dtype=torch.float32)

        self.ys['optimal'] = torch.from_numpy(self.y_pred_optimal).to(dtype=torch.float32)
        self.ys['linear'] = torch.from_numpy(self.y_linear).to(dtype=torch.float32)
        self.ys['shift'] = torch.from_numpy(self.y_shift).to(dtype=torch.float32)
        self.ys['seasonal'] = torch.from_numpy(self.y_seasonal).to(dtype=torch.float32)
        self.ys['conditional'] = torch.from_numpy(self.y_conditional).to(dtype=torch.float32)
        self.ys['cs'] = torch.from_numpy(self.y_cs).to(dtype=torch.float32)
        self.ys['cs_shift'] = torch.from_numpy(self.y_cs_shift).to(dtype=torch.float32)
        self.ys['true'] = torch.from_numpy(y).to(dtype=torch.float32)

    def get_dataloader(self, n_rolling=1, narrow=False, train_test_split=True, shuffle=True, batch_size=256):
        if self.X is None:
            raise UserWarning('Dataset not generated yet, please run generate_dataset() first')

        if (narrow and n_rolling > 1 and not self.is_transposed):
            for name in self.ys:
                self.ys[name] = self.ys[name].transpose(0, 1)
            self.is_transposed = True
        elif self.is_transposed and (not narrow or n_rolling == 1):
            for name in self.ys:
                self.ys[name] = self.ys[name].transpose(0, 1)
            self.is_transposed = False


        if train_test_split:
            self.train, self.test = utils.np_to_torch(self.X, self.y, n_rolling=n_rolling, narrow=narrow,
                                                      train_test_split=train_test_split, shuffle=shuffle,
                                                      batch_size=batch_size)
        else:
            self.train = utils.np_to_torch(self.X, self.y, n_rolling=n_rolling, narrow=narrow,
                                           train_test_split=train_test_split, shuffle=shuffle,
                                           batch_size=batch_size)
