import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.datasets import load_iris
from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here

n_estimators = 3
iris = load_iris()
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for pair in ([0, 1], [0, 2], [2, 3]):
    X = pd.DataFrame(iris.data[:,pair])
    y = pd.Series(iris.target,dtype="category")
    NUM_OP_CLASSES = len(np.unique(y))
    criteria = "entropy"
    Classifier_AB = RandomForestClassifier(n_estimators=n_estimators,criterion=criteria)
    Classifier_AB.fit(X, y)
    y_hat = Classifier_AB.predict(X)
    #print(y_hat)
    [fig1,fig2] = Classifier_AB.plot(X,y,True,pair)
    for label in fig1.get_axes():
        label.set_xlabel(f"{features[pair[0]]}")
        label.set_ylabel(f"{features[pair[1]]}")
    for label in fig2.get_axes():
        label.set_xlabel(f"{features[pair[0]]}")
        label.set_ylabel(f"{features[pair[1]]}")
    if pair==[0,1]:
        fig1.savefig(fname='sepal length sepal width.png')
        fig2.savefig(fname='sepal length sepal width Common.png')
    elif pair==[0,2]:
        fig1.savefig(fname='sepal length petal length.png')
        fig2.savefig(fname='sepal length petal length Common.png')
    else:
        fig1.savefig(fname='petal length petal width.png')
        fig2.savefig(fname='petal length petal width Common.png')
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    #print(y_hat)
"""
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
"""