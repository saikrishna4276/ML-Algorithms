import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from metrics import *
from sklearn.datasets import make_classification
from ensemble.bagging import BaggingClassifier
from tree.base import WeightedDecisionTree
import multiprocessing
# Or use sklearn decision tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
########### BaggingClassifier ###################

X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

X = pd.DataFrame(X)
y = pd.Series(y,dtype='category')
n_estimators = 5
tree = DecisionTreeClassifier


for i in ['single','parallel']:
    Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators,n_jobs=i)
    start_time = time.perf_counter()
    duration = Classifier_B.fit(X, y)
    finish_time = time.perf_counter()
    y_hat = Classifier_B.predict(X)
    print("Accuracy: ", accuracy(y_hat, y))
    '''for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))'''
    [fig1, fig2] = Classifier_B.plot(X,y)
    if i=='single':
        print(f'Regular Implementation(time) for n_jobs = {i}: ',finish_time-start_time
    )
        fig1.savefig(fname="Q4_single_Fig1.png")
        fig2.savefig(fname= "Q4_single_Fig2.png")
    else:
        print(f'Parallel Implementation(time) for n_jobs = {i}: ',finish_time-start_time
    )
        fig1.savefig(fname="Q4_parallel_Fig1.png")
        fig2.savefig(fname= "Q4_parallel_Fig2.png")

    
    