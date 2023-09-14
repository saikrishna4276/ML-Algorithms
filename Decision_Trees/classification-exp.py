import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read dataset
# ...
# 
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X_data = pd.DataFrame(X)
y_data = pd.Series(y,dtype="category")
split = int(0.7 * len(X))
X_train = X_data[:split]
X_test = X_data[split:]
y_train = y_data[:split]
y_test = y_data[split:]
#print(X)
#print(X_train)
#print(y_test)
# For plotting
for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain and gini_index
    tree.fit(X_train, y_train)
    y_train_hat = tree.predict(X_train)
    y_hat = tree.predict(X_test)
    #print(y_hat)
    tree.plot()
    print('Criteria :', criteria)
    print('----->Training Accuracy: ', accuracy(y_train_hat, y_train))
    print('----->Testing Accuracy: ', accuracy(y_hat, y_test))
    for i in set(y):
        print("Class =",i)
        print('----->Precision: ', precision(y_hat, y_test, i))
        print('----->Recall: ', recall(y_hat, y_test, i))
#import matplotlib.pyplot as plt
#plt.scatter(X[:, 0], X[:, 1], c=y)

best_acc = 0
for i in range(5):
    split_1 = int((i/5)*len(X))
    split_2 = int(((i+1)/5)*len(X))
    X = pd.concat([X_data.iloc[:split_1, :], X_data.iloc[split_2:,:]], ignore_index=True)
    X_test = X_data.iloc[split_1:split_2, :]
    y = pd.concat([y_data.iloc[:split_1], y_data.iloc[split_2:]], ignore_index=True)
    y_test = y_data.iloc[split_1:split_2]
    tree = DecisionTree(criterion="information_gain", max_depth=3)
    tree.fit(X, y)
    y_test_hat = tree.predict(X_test)
    best_acc += accuracy(y_test_hat, y_test)

print("Average accuracy for 5 fold cross-validation:", best_acc/5)


"""
Finding Optimal depth using nested cross-validation
"""

Accuracy_plot = []              # list used to plot validation accuracy

outer = 5                   #  5 fold cross-validation
inner = 4               # no. of folds in nested cross-validaton

best_acc = -1
Optimal_Depth = -1
for i in range(outer):
    split_1 = int((i/outer)*len(X))
    split_2 = int(((i+1)/outer)*len(X))
    X = pd.concat([X_data.iloc[:split_1, :], X_data.iloc[split_2:,:]], ignore_index=True)
    X_test = X_data.iloc[split_1:split_2, :]
    y = pd.concat([y_data.iloc[:split_1], y_data.iloc[split_2:]], ignore_index=True)
    y_test = y_data.iloc[split_1:split_2]
    Metrics = {"accuracy": -1, "depth":-1}
    val_accuracy = []
    for depth in range(10):
        temp = 0
        for j in range(inner):
            split_1 = int((j/inner)*X.shape[0])
            split_2 = int(((j+1)/inner)*X.shape[0])
            X_val = X.iloc[split_1:split_2, :]
            X_train = pd.concat([X.iloc[:split_1, :], X.iloc[split_2:, :]],ignore_index=True)
            y_val = y.iloc[split_1:split_2]
            y_train = pd.concat([y.iloc[:split_1], y.iloc[split_2:]],ignore_index=True)
            tree = DecisionTree(criterion="information_gain", max_depth=depth)
            tree.fit(X_train, y_train)
            y_val_pred = tree.predict(X_val)
            temp += accuracy(y_val_pred, y_val)
        temp = temp/inner
        if(Metrics["accuracy"]==-1):
            Metrics["accuracy"] = temp
            Metrics["depth"] = depth
        else:
            if(temp>Metrics["accuracy"]):
                Metrics["accuracy"] = temp
                Metrics["depth"] = depth
        val_accuracy.append(temp)
    Accuracy_plot.append(val_accuracy)
    tree = DecisionTree(criterion="information_gain", max_depth=Metrics["depth"])
    tree.fit(X,y)
    y_test_hat = tree.predict(X_test)
    temp_acc = accuracy(y_test_hat, y_test)
    if(temp_acc>best_acc):
        best_acc = temp_acc
        Optimal_Depth = Metrics["depth"]
    print(f"Accuracy for outer fold {i+1}:",temp_acc, "depth:", Metrics["depth"])



print("Optimal Depth:", Optimal_Depth)               # Showing Optimal Depth for best results


# Plot results of 5 folds just for comparision of depth vs accuracy

depth_array = [[i for i in range(10)] for j in range(outer)]

fig = plt.figure()
depth_acc = plt.subplot()
for i in range(outer):
    depth_acc.plot(depth_array[i], Accuracy_plot[i],label = "outer"+str(i+1))
depth_acc.set_xlabel('depth')
depth_acc.set_ylabel('accuracy')
plt.legend()
plt.show()