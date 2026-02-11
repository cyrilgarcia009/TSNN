# Time-series forecasting using transformers.

In this repo we test the effectiveness of transformer models for the problem of multi-dimensional time-series forecasting, in the setting where the signal to noise ratio is small. The setup is to predict a Y series of dimensions (time_steps, n_var) from an X of dimensions (time_steps, n_var, n_fea). The attention mechanism will be implemented both in the time-series and in the cross-sectional (given by the n_var variables) directions. We test our models on synthetic noisy data, including simple examples of linear / nonlinear relationships and conditionings, in both the time series and cross-sectional dimensions. We compare the predictive power of the model to the simple benchmarks of linear and Lasso regresions, boosting, and of simple neural networks architectures.

Our paper: [arXiv:2602.09869](https://arxiv.org/abs/2602.09869).

## Contributors

- [Cyril Garcia](https://github.com/cyrilgarcia009)
- [Guillaume Remy](https://github.com/GuillaumeRemy92)
