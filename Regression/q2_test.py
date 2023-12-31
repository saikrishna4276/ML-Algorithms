# -*- coding: utf-8 -*-
"""Q2_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R-h6mR7sy_h7Yi1dKE4xCb8nKprgkVDj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linear_regression import LinearRegression
from sklearn.metrics import mean_squared_error as mse

np.random.seed(45)

N = 90
P = 10
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))





LR = LinearRegression(fit_intercept=True)

# Call Gradient Descent here
print("Gradient Type: Manual")
LR.fit_gradient_descent(X, y, batch_size=X.shape[0], gradient_type='manual', penalty_type = 'unregularized', penalty_value=0.05,num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--Unregularized rmse: ", np.sqrt(mse(y,y_hat)))

LR.fit_gradient_descent(X, y, batch_size=X.shape[0], gradient_type='manual', penalty_type = 'l2', penalty_value=0.05,num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--L2 rmse: ", np.sqrt(mse(y,y_hat)))

print("---------------------------")

print("Gradient Type: JAX")
LR.fit_gradient_descent(X, y, batch_size=X.shape[0], gradient_type='jax', penalty_type = 'unregularized', penalty_value=0.05,num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--Unregularized rmse: ", np.sqrt(mse(y,y_hat)))

LR.fit_gradient_descent(X, y, batch_size=X.shape[0], gradient_type='jax', penalty_type = 'l1', penalty_value=0.05,num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--L1 rmse: ", np.sqrt(mse(y,y_hat)))

LR.fit_gradient_descent(X, y, batch_size=X.shape[0], gradient_type='jax', penalty_type = 'l2', penalty_value=0.05, num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--L2 rmse: ", np.sqrt(mse(y,y_hat)))

print("---------------------------")

print("Running SGD with ridge regularization")

LR.fit_gradient_descent(X, y, batch_size=1, gradient_type='manual', penalty_type = 'l2', penalty_value=0.05, num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--rmse: ", np.sqrt(mse(y,y_hat)))

print("---------------------------")

print("Running mini-batch SGD with ridge regularization")

LR.fit_gradient_descent(X, y, batch_size=10, gradient_type='manual', penalty_type = 'l2', penalty_value=0.05, num_iters=10, lr=0.01)
y_hat = LR.predict(X)
print("--rmse: ", np.sqrt(mse(y,y_hat)))


print("Running SGD with momentum and ridge regularization")
LR.fit_SGD_with_momentum(X, y, gradient_type='manual', penalty_type = 'l2', penalty_value=1.25,num_iters=40, lr=0.01, beta=0.05)
y_hat = LR.predict(X)
print("--rmse: ", np.sqrt(mse(y,y_hat)))


