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

    def correl(self, dataloader, generator):
        """
        Correl matrix of all models, true y and optimal prediction
        :param X: data to predict
        :return:
        """
        X, y_true = utils.torch_to_np(dataloader)
        res = [pd.Series(y_true).rename('y_true')]
        # if y_optimal is not None:
        loader_idx = dataloader.dataset.indices
        if len(loader_idx) > generator.y_pred_optimal.shape[0]:
            res.append(pd.Series(generator.y_pred_optimal.flatten()[loader_idx]).rename('y_optimal'))
            res.append(pd.Series(generator.y_linear.flatten()[loader_idx]).rename('y_linear'))
            res.append(pd.Series(generator.y_seasonal.flatten()[loader_idx]).rename('y_seasonal'))
            res.append(pd.Series(generator.y_conditional.flatten()[loader_idx]).rename('y_conditional'))
            res.append(pd.Series(generator.y_shift.flatten()[loader_idx]).rename('y_shift'))
        else:
            res.append(pd.Series(generator.y_pred_optimal[loader_idx].flatten()).rename('y_optimal'))
            res.append(pd.Series(generator.y_linear[loader_idx].flatten()).rename('y_linear'))
            res.append(pd.Series(generator.y_seasonal[loader_idx].flatten()).rename('y_seasonal'))
            res.append(pd.Series(generator.y_conditional[loader_idx].flatten()).rename('y_conditional'))
            res.append(pd.Series(generator.y_shift[loader_idx].flatten()).rename('y_shift'))

        for k in range(len(self.models)):
            res.append(pd.Series(self.models[k].predict(dataloader)).rename(self.model_names[k]))
        res = pd.concat(res, axis=1).corr()
        upper_tri_idx = np.triu(np.ones(res.shape)).astype(bool)
        res = res.mask(upper_tri_idx).dropna(how='all', axis=0).dropna(how='all', axis=1)
        return res.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=4)
