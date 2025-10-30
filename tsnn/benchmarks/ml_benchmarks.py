from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
import numpy as np
from .. import utils


class LassoBenchmark:
    def __init__(self, alphas=100):
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
