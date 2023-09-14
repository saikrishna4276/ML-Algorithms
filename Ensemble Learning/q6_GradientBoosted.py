import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.metrics import mean_squared_error
from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import WeightedDecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import sklearn.ensemble
from metrics import rmse
X, y= make_regression(
       n_features=3,
       n_informative=3,
       noise=10,
       tail_strength=10,
       random_state=42,
   )

X = pd.DataFrame(X)
y = pd.Series(y)
#construction of Gradient boosting using Sklearn decision tree

tree = DecisionTreeRegressor
for estimators in range(100,1001,100):
    GradBoost = GradientBoostedRegressor(tree, n_estimators=estimators, learning_rate =0.1, max_depth=2)
    GradBoost.fit(X,y)
    y_hat = GradBoost.predict(X)
    print('No. of estimators:',estimators)
    print(f'MSE of Gradient boosting implemented is {mean_squared_error(y,y_hat)}')
    GradBoost_sk = sklearn.ensemble.GradientBoostingRegressor(max_depth=2,learning_rate=0.1,n_estimators=estimators,loss="squared_error")
    GradBoost_sk.fit(X,y)
    y_hat = GradBoost_sk.predict(X)
    print(f'MSE of Gradient boosting from sklearn is {mean_squared_error(y,y_hat)}')
