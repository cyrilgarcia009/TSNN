from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
import numpy as np

class LassoBenchmark:
    def __init__(self, alphas=100):
        self.alphas = alphas
        self.model = LassoCV(alphas=alphas)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return np.corrcoef(self.model.predict(X), y)[0][1]

class CustomBenchmark:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return np.corrcoef(self.model.predict(X), y)[0][1]
