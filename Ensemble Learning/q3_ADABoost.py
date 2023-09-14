import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.model_selection import train_test_split
from ensemble.ADABoost import AdaBoostClassifier
#from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
# Or you could import sklearn DecisionTree
from sklearn.datasets import load_iris
np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y,dtype='category')
n_estimators = 3
NUM_OP_CLASSES = len(np.unique(y))
criteria = "entropy"
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators, classes=NUM_OP_CLASSES, max_depth=1)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
#print(y_hat)
[fig1, fig2, fig] = Classifier_AB.plot(X,y)
fig1.savefig(fname=f"Adaboost for individual estimator")
fig2.savefig(fname=f"Adaboost for combined")
fig.savefig(fname=f"Alphas for adaboost")
print("Criteria :", criteria)
print(f"Accuracy of implemented adaboost model for: {accuracy_score(y,y_hat)}")
#print(y_hat)

from sklearn.ensemble import AdaBoostClassifier
base = DecisionTreeClassifier(max_depth=1)
Classifier_AB = AdaBoostClassifier(estimator =base,n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
print(f"Accuracy of sklearn model: {accuracy_score(y,y_hat)}")

exit()