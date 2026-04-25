from sklearn.linear_model import Lasso, LassoCV, Ridge
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


class GlobalXSummaryLinearBenchmark:
    def __init__(self, alpha: float = 1.0):
        """
        Linear benchmark on summarized rolling X windows.
        Uses a compact set of lag summaries instead of the full flattened lag stack.
        """
        self.model = Ridge(alpha=alpha)

    def _build_features(self, dataloader):
        if not hasattr(dataloader.dataset, 'indices') or not hasattr(dataloader.dataset.dataset, 'X'):
            raise ValueError("GlobalXSummaryLinearBenchmark requires a DataLoader backed by a Subset with X values.")

        indices = list(dataloader.dataset.indices)
        dataset = dataloader.dataset.dataset

        X_full = dataset.X
        if hasattr(X_full, "detach"):
            X_full = X_full.detach().cpu().numpy()
        else:
            X_full = np.asarray(X_full)

        if X_full.ndim != 3:
            raise ValueError(
                f"GlobalXSummaryLinearBenchmark expects X of shape (T, n_ts, n_f), got {X_full.shape}."
            )

        y_full = None
        if hasattr(dataset, 'y'):
            y_full = dataset.y
            if hasattr(y_full, "detach"):
                y_full = y_full.detach().cpu().numpy()
            else:
                y_full = np.asarray(y_full)

        X_rows = []
        y_rows = []

        for idx in indices:
            window = X_full[:idx + 1]
            last = window[-1]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            if window.shape[0] > 1:
                slope = (window[-1] - window[0]) / (window.shape[0] - 1)
            else:
                slope = np.zeros_like(last)

            feature_vec = np.concatenate(
                [last.reshape(-1), mean.reshape(-1), std.reshape(-1), slope.reshape(-1)],
                axis=0,
            )
            X_rows.append(feature_vec)
            if y_full is not None:
                y_rows.append(y_full[idx])

        X = np.stack(X_rows, axis=0)
        if y_full is None:
            return X

        y = np.stack(y_rows, axis=0)
        return X, y

    def fit(self, dataloader):
        X, y = self._build_features(dataloader)
        self.model.fit(X, y)

    def predict(self, dataloader):
        built = self._build_features(dataloader)
        X = built[0] if isinstance(built, tuple) else built
        pred = self.model.predict(X)
        return pred.reshape(-1)

    def score(self, dataloader):
        X, y = self._build_features(dataloader)
        pred = self.model.predict(X)
        return np.corrcoef(pred.reshape(-1), y.reshape(-1))[0][1]


class RollingWindowLinearBenchmark:
    def __init__(
        self,
        model_type: str = "ridge",
        alpha_grid=None,
        n_rolling: int = 10,
        val_pct: float = 0.2,
        selection_metric: str = "mse",
        max_iter: int = 10000,
    ):
        """
        Linear benchmark on the full rolling window used by the neural models.

        Each target series is treated as one supervised example per date. For a
        target series i, the cross-section is cyclically reordered so series i is
        first, matching the older global-linear convention while preserving a
        fixed feature dimension across targets.
        """
        if model_type not in {"ridge", "lasso"}:
            raise ValueError("model_type must be 'ridge' or 'lasso'.")
        if selection_metric not in {"mse", "corr"}:
            raise ValueError("selection_metric must be 'mse' or 'corr'.")

        self.model_type = model_type
        self.n_rolling = n_rolling
        self.val_pct = val_pct
        self.selection_metric = selection_metric
        self.max_iter = max_iter
        if alpha_grid is None:
            if model_type == "ridge":
                alpha_grid = np.logspace(-4, 3, 15)
            else:
                alpha_grid = np.logspace(-5, -1, 9)
        self.alpha_grid = list(alpha_grid)
        self.best_alpha_ = None
        self.model = None

    def _new_model(self, alpha):
        if self.model_type == "ridge":
            return Ridge(alpha=float(alpha))
        return Lasso(alpha=float(alpha), max_iter=self.max_iter)

    def _build_features(self, dataloader):
        if not hasattr(dataloader.dataset, "indices") or not hasattr(dataloader.dataset.dataset, "X"):
            raise ValueError("RollingWindowLinearBenchmark requires a DataLoader backed by a Subset with X values.")

        indices = list(dataloader.dataset.indices)
        dataset = dataloader.dataset.dataset

        X_full = dataset.X
        if hasattr(X_full, "detach"):
            X_full = X_full.detach().cpu().numpy()
        else:
            X_full = np.asarray(X_full)

        if X_full.ndim != 3:
            raise ValueError(
                f"RollingWindowLinearBenchmark expects X of shape (T, n_ts, n_f), got {X_full.shape}."
            )

        y_full = None
        if hasattr(dataset, "y"):
            y_full = dataset.y
            if hasattr(y_full, "detach"):
                y_full = y_full.detach().cpu().numpy()
            else:
                y_full = np.asarray(y_full)

        _, n_ts, n_f = X_full.shape
        zero_panel = np.zeros((n_ts, n_f), dtype=X_full.dtype)
        X_rows = []
        y_rows = []

        for idx in indices:
            lag_panels = []
            for lag in range(self.n_rolling):
                past_idx = idx - lag
                if past_idx >= 0:
                    lag_panels.append(X_full[past_idx])
                else:
                    lag_panels.append(zero_panel)

            for target_i in range(n_ts):
                feature_parts = []
                for panel in lag_panels:
                    reordered = np.concatenate([panel[target_i:], panel[:target_i]], axis=0)
                    feature_parts.append(reordered.reshape(-1))
                X_rows.append(np.concatenate(feature_parts, axis=0))
                if y_full is not None:
                    y_rows.append(y_full[idx, target_i])

        X = np.stack(X_rows, axis=0)
        if y_full is None:
            return X
        return X, np.asarray(y_rows)

    def _score_validation(self, pred, y):
        pred = np.asarray(pred).reshape(-1)
        y = np.asarray(y).reshape(-1)
        finite = np.isfinite(pred) & np.isfinite(y)
        pred = pred[finite]
        y = y[finite]
        if self.selection_metric == "mse":
            return float(np.mean((pred - y) ** 2))
        if pred.size < 2 or np.std(pred) == 0 or np.std(y) == 0:
            return -np.inf
        return float(np.corrcoef(pred, y)[0, 1])

    def fit(self, dataloader):
        X, y = self._build_features(dataloader)
        if self.val_pct and self.val_pct > 0:
            n = X.shape[0]
            n_val = max(1, int(n * self.val_pct))
            if n_val >= n:
                raise ValueError("val_pct leaves no samples for fitting.")
            X_fit, y_fit = X[:-n_val], y[:-n_val]
            X_val, y_val = X[-n_val:], y[-n_val:]

            best_score = None
            best_alpha = None
            for alpha in self.alpha_grid:
                model = self._new_model(alpha)
                model.fit(X_fit, y_fit)
                score = self._score_validation(model.predict(X_val), y_val)
                is_better = (
                    best_score is None
                    or (self.selection_metric == "mse" and score < best_score)
                    or (self.selection_metric == "corr" and score > best_score)
                )
                if is_better:
                    best_score = score
                    best_alpha = alpha
            self.best_alpha_ = best_alpha
        else:
            self.best_alpha_ = self.alpha_grid[0]

        self.model = self._new_model(self.best_alpha_)
        self.model.fit(X, y)

    def predict(self, dataloader):
        if self.model is None:
            raise ValueError("Model must be fit before predict.")
        built = self._build_features(dataloader)
        X = built[0] if isinstance(built, tuple) else built
        return self.model.predict(X).reshape(-1)

    def score(self, dataloader):
        X, y = self._build_features(dataloader)
        pred = self.model.predict(X)
        return np.corrcoef(pred.reshape(-1), y.reshape(-1))[0][1]


class GlobalVARBenchmark:
    def __init__(self, n_lags: int = 10, alpha: float = 1.0):
        """
        Restricted global VAR-style linear benchmark on lagged targets only.
        Uses previous target values across all series as predictors for all series.
        """
        self.n_lags = n_lags
        self.model = Ridge(alpha=alpha)

    def _build_features(self, dataloader):
        if not hasattr(dataloader.dataset, 'indices') or not hasattr(dataloader.dataset.dataset, 'y'):
            raise ValueError("GlobalVARBenchmark requires a DataLoader backed by a Subset with target values.")

        indices = list(dataloader.dataset.indices)
        y_full = dataloader.dataset.dataset.y
        if hasattr(y_full, "detach"):
            y_full = y_full.detach().cpu().numpy()
        else:
            y_full = np.asarray(y_full)

        if y_full.ndim != 2:
            raise ValueError(f"GlobalVARBenchmark expects 2D targets of shape (T, n_ts), got {y_full.shape}.")

        T, n_ts = y_full.shape
        X_rows = []
        y_rows = []
        for idx in indices:
            lagged = []
            for lag in range(1, self.n_lags + 1):
                past_idx = idx - lag
                if past_idx >= 0:
                    lagged.append(y_full[past_idx])
                else:
                    lagged.append(np.zeros(n_ts, dtype=y_full.dtype))
            X_rows.append(np.concatenate(lagged, axis=0))
            y_rows.append(y_full[idx])

        X = np.stack(X_rows, axis=0)
        y = np.stack(y_rows, axis=0)
        return X, y

    def fit(self, dataloader):
        X, y = self._build_features(dataloader)
        self.model.fit(X, y)

    def predict(self, dataloader):
        X, _ = self._build_features(dataloader)
        pred = self.model.predict(X)
        return pred.reshape(-1)

    def score(self, dataloader):
        X, y = self._build_features(dataloader)
        pred = self.model.predict(X)
        return np.corrcoef(pred.reshape(-1), y.reshape(-1))[0][1]
