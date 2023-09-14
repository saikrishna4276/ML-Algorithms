
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree

np.random.seed(42)

# Read auto-mpg.data
# ...
# 
names = ['mpg', 'cylinders', 'displacement','horsepower', 'weight', 'acceleration', 'model year', 'origin','car name']
data = pd.read_csv('auto-mpg.data', sep = '\s+', header = None, names = names)

data_clean=data.applymap(lambda x: np.nan if x == '?' else x).dropna()

data_clean["horsepower"] = pd.to_numeric(data_clean["horsepower"])
data_clean["cylinders"]=pd.Series(data_clean['cylinders'],dtype="category")
data_clean["origin"]=pd.Series(data_clean['origin'],dtype="category")
data_clean=data_clean.reset_index(drop=True)

# 70:30 train test split
train_test_split = int(0.7*data.shape[0])
maxdepth=4
X = data_clean.iloc[:train_test_split, 1:-1]
X_test = data_clean.iloc[train_test_split:, 1:-1]
y = pd.Series(data_clean.iloc[:train_test_split, 0])
y_test = pd.Series(data_clean.iloc[train_test_split:, 0])
#print(X)
#print(y)
"""for i in names:
    print(type(data[i][0]))"""
# Building Decesion Tree based on my model
criteria = 'information_gain'
mytree = DecisionTree(criterion=criteria, max_depth=maxdepth) #Split based on Inf. Gain
mytree.fit(X, y)
mytree.plot()

print("My Model")
y_hat = mytree.predict(X)
print("Train Scores:")
print('\tRMSE: ', rmse(y_hat, y))
print('\tMAE: ', mae(y_hat, y))

y_test_hat = mytree.predict(X_test)
print("Test Scores:")
print('\tRMSE: ', rmse(y_test_hat, y_test))
print('\tMAE: ', mae(y_test_hat, y_test))


"""
Decision Tree using SKlearn
"""

# Building Decesion Tree based on sklearn
print("Sklearn Model")
clf = tree.DecisionTreeRegressor(max_depth=maxdepth) #using Regressor as the output response is continous or numerical
clf = clf.fit(X,y)
y_test_hat = pd.Series(clf.predict(X_test))
print("Test Scores:")
print('\tRMSE: ', rmse(y_test_hat, y_test))
print('\tMAE: ', mae(y_test_hat, y_test))
