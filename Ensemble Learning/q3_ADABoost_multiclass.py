import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
#from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or you could import sklearn DecisionTree
from sklearn.datasets import load_iris
np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


from sklearn.metrics import accuracy_score

n_estimators = 3
iris = load_iris()
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for pair in ([0, 1], [0, 2], [2, 3]):
    X = pd.DataFrame(iris.data[:,pair])
    y = pd.Series(iris.target,dtype="category")
    #print(X)
    NUM_OP_CLASSES = len(np.unique(y))
    criteria = "entropy"
    tree = DecisionTreeClassifier
    Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators, classes=NUM_OP_CLASSES, max_depth=2)
    Classifier_AB.fit(X, y)
    y_hat = Classifier_AB.predict(X)
    #print(y_hat)
    [fig1, fig2, fig] = Classifier_AB.plot(X,y)
    for label in fig1.get_axes():
        label.set_xlabel(f"{features[pair[0]]}")
        label.set_ylabel(f"{features[pair[1]]}")
    for label in fig2.get_axes():
        label.set_xlabel(f"{features[pair[0]]}")
        label.set_ylabel(f"{features[pair[1]]}")
    for label in fig.get_axes():
        label.set_xlabel(f"{features[pair[0]]}")
        label.set_ylabel(f"{features[pair[1]]}")
    fig1.savefig(fname=f"multi class adaboost for feature {features[pair[0]]} {features[pair[1]]}")
    fig2.savefig(fname=f"multi class common adaboost for feature {features[pair[0]]} {features[pair[1]]}")
    fig.savefig(fname=f"multi class adaboost for alphas {features[pair[0]]} {features[pair[1]]}")
    print("Criteria :", criteria)
    print(f"Accuracy of implemented adaboost model for: {accuracy_score(y,y_hat)}")
    #print(y_hat)
    break
from sklearn.ensemble import AdaBoostClassifier

for pair in ([0, 1], [0, 2], [2, 3]):
    X = pd.DataFrame(iris.data[:,pair])
    y = pd.Series(iris.target)
    tree = DecisionTreeClassifier(max_depth = 2)
    Classifier_AB = AdaBoostClassifier(estimator = tree,n_estimators=n_estimators)
    Classifier_AB.fit(X, y)
    y_hat = Classifier_AB.predict(X)
    print(f"Accuracy of sklearn model: {accuracy_score(y,y_hat)}")
    break
exit()
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))

