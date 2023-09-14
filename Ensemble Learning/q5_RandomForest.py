import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")


Classifier_RF = RandomForestClassifier(10)
Classifier_RF.fit(X, y)
y_hat = Classifier_RF.predict(X)
Classifier_RF.plot(X,y)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print(f'Class: {cls}')
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

Regressor_RF = RandomForestRegressor(10)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
Regressor_RF.plot()
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
