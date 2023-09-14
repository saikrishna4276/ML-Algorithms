
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import seaborn as sns
import seaborn as sns
np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
def createFakeData(N,M,case):
    if(case==1):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif(case==2):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    elif(case==3):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))

    return X, y


def time_plot(timeData,case,type):
    df = pd.DataFrame(data=timeData)
    heatmap = pd.pivot_table(df, values='time', index=['N'], columns='M')
    sns.heatmap(heatmap, cmap="flare")
    if case==1:
        plt.savefig("r_r"+" "+type+".png")
    elif case == 2:
        plt.savefig("r_d"+" "+type+".png")
    elif case ==3:
        plt.savefig("d_d"+" "+type+".png")
    else:
        plt.savefig("d_r"+" "+type+".png")
    plt.clf()


def calculate_time(case):
    assert(1<=case<=4)
    train_time = {'N':[], 'M':[], 'time':[]}
    test_time = {'N':[], 'M':[], 'time':[]}
    for N in range(45,55):
        for M in range(2,10):
            X, y = createFakeData(N,M,case)
            tree = DecisionTree(criterion="information_gain", max_depth=3)
            #train data
            startTime = time.time()
            tree.fit(X,y)
            endTime = time.time()
            train_time['N'].append(N)
            train_time['M'].append(M)
            train_time['time'].append(endTime - startTime)
            #test data
            startTime = time.time()
            y_hat = tree.predict(X)
            endTime = time.time()
            test_time['N'].append(N)
            test_time['M'].append(M)
            test_time['time'].append(endTime - startTime)
    
    time_plot(train_time,case,'train')
    time_plot(test_time,case,'test')


calculate_time(3)
print("Discrete Input and Discrete Output plot done!!!")
calculate_time(2)
print("Real Input and Discrete Output plot done!!!")
calculate_time(4)
print("Discrete Input and Real Output plot done!!!")
calculate_time(1)
print("Real Input and Real Output plot done!!!")