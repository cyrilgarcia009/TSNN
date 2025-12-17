from sklearn.linear_model import LassoCV
import numpy as np
from .. import utils


class LassoBenchmark:
    def __init__(self, alphas=100):
        """
        A simple cross-validated lasso benchmark
        :param alphas:
        """
        self.alphas = alphas
        self.model = LassoCV(alphas=alphas)

    def fit(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        self.model.fit(X, y)

    def predict(self, dataloader):
        if hasattr(dataloader.dataset.dataset, 'y'):
            X, y = utils.torch_to_np(dataloader)
        else:
            X = utils.torch_to_np(dataloader)
        return self.model.predict(X)

    def score(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        return np.corrcoef(self.model.predict(X), y)[0][1]


class CustomBenchmark:
    def __init__(self, model):
        """
        Generic class to be used with any model with fit and predict methods
        :param model: model to use, for instance LinearRegression()
        """
        self.model = model

    def fit(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        self.model.fit(X, y)

    def predict(self, dataloader):
        if hasattr(dataloader.dataset.dataset, 'y'):
            X, y = utils.torch_to_np(dataloader)
        else:
            X = utils.torch_to_np(dataloader)
        return self.model.predict(X)

    def score(self, dataloader):
        X, y = utils.torch_to_np(dataloader)
        return np.corrcoef(self.model.predict(X), y)[0][1]


class CustomBenchmarkRolling:
    def __init__(self, model, n_rolling: int = 10):
        """
        Greedy model which puts all features, all time series and all lags as variables considered
        :param model: model to use, for instance LinearRegression()
        :param n_rolling: number of time steps to consider in the model
        """
        self.model = model
        self.n_rolling = n_rolling

    def fit(self, dataloader):
        X, y = utils.torch_to_np_features(dataloader, n_rolling=self.n_rolling)
        self.model.fit(X, y)

    def predict(self, dataloader):
        if hasattr(dataloader.dataset.dataset, 'y'):
            X, y = utils.torch_to_np_features(dataloader, n_rolling=self.n_rolling)
        else:
            X = utils.torch_to_np_features(dataloader, n_rolling=self.n_rolling)
        return self.model.predict(X)

    def score(self, dataloader):
        X, y = utils.torch_to_np_features(dataloader, n_rolling=self.n_rolling)
        return np.corrcoef(self.model.predict(X), y)[0][1]
