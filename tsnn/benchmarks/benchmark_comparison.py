import pandas as pd

class Comparator:
    def __init__(self, models: list, model_names=None):
        self.models = models
        if model_names is None:
            self.model_names = [f'model_{k}' for k in range(len(models))]
        elif len(model_names) != len(models):
            raise UserWarning('len of models and model names must be equal')
        else:
            self.model_names = model_names


    def correl(self, X, y_true, y_optimal):
        """
        Correl matrix of all models, true y and optimal prediction
        :param X: data to predict
        :param y_true: true y
        :param y_optimal: optimal y (computed in generator)
        :return:
        """
        res = [pd.Series(y_true).rename('y_true'), pd.Series(y_optimal).rename('y_optimal')]
        for k in range(len(self.models)):
            res.append(pd.Series(self.models[k].predict(X)).rename(self.model_names[k]))
        return pd.concat(res, axis=1).corr()


