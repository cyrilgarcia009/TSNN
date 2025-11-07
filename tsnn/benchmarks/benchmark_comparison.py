import pandas as pd
import numpy as np
from .. import utils


class Comparator:
    def __init__(self, models: list, model_names=None):
        self.models = models
        if model_names is None:
            self.model_names = [f'model_{k}' for k in range(len(models))]
        elif len(model_names) != len(models):
            raise UserWarning('len of models and model names must be equal')
        else:
            self.model_names = model_names

    def correl(self, generator, mode='train'):
        """
        Correl matrix of all models, true y and optimal prediction
        :param X: data to predict
        :return:
        """
        if mode == 'train':
            dataloader = generator.train
        else:
            dataloader = generator.test

        loader_idx = dataloader.dataset.indices
        res = []
        if len(loader_idx) > generator.ys['true'].shape[0]:
            for name in generator.ys:
                res.append(pd.Series(generator.ys[name].flatten()[loader_idx]).rename(name))
        else:
            for name in generator.ys:
                res.append(pd.Series(generator.ys[name][loader_idx].flatten()).rename(name))

        for k in range(len(self.models)):
            prediction = pd.Series(self.models[k].predict(dataloader)).rename(self.model_names[k])
            if len(prediction) != len(res[0]):
                raise UserWarning(
                    f'data dimension mismatch for model {self.model_names[k]}, the dataloader object must match the prediction model')
            res.append(prediction)
        res = pd.concat(res, axis=1).corr()

        # setting upper triangle to NaN for better visualization
        upper_tri_idx = np.triu(np.ones(res.shape)).astype(bool)
        res = res.mask(upper_tri_idx).dropna(how='all', axis=0).dropna(how='all', axis=1)
        return res.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=4)
