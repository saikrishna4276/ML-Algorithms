import numpy as np
import pandas as pd
#from tree.base import DecisionTree
from sklearn import tree
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
random_seed=np.random.seed(1234)
def normalize(l):
    result = l[:]
    maxi = max(result)
    mini = min(result)
    for i in range(len(result)):
        result[i] = (result[i]-mini)/(maxi-mini)
    return result
def Regression():    
    np.random.seed(1234)
    x = np.linspace(0, 10, 50)
    eps = np.random.normal(0, 5, 50)
    y = pd.Series(x**2 + 1 + eps)
    data = pd.DataFrame({'x':x,'eps':eps})  
    return data, y
def Classification():
    X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
    X = pd.DataFrame(X)
    y = pd.Series(y,dtype='category')
    return X, y
def test_train(X,y):
    train_test_split = int(0.7*X.shape[0])
    X_train = X.iloc[:train_test_split,:]
    X_test = X.iloc[train_test_split:,:]
    y_train = y.iloc[:train_test_split]
    y_test = y.iloc[train_test_split:] 
    return X_train,X_test,y_train,y_test 
def bias_variance(X_train, X_test,y_train,y_test,depths,number_of_rounds=100):
    bias_plot=[]
    variance_plot=[]
    if y_test.dtype == "category":
        dtype = np.int64
        flag = True
        random=1234
    else:
        dtype = np.float64
        flag =False
        random=1234
    various_models = np.zeros((number_of_rounds,y_test.shape[0]),dtype=dtype)
   # out = pd.DataFrame(columns = ['Bias^2', 'Variance'])
    bias_plot = []
    variance_plot = []

    for depth in range(1,depths):
        for rounds in range(number_of_rounds):
            if flag:
                base = tree.DecisionTreeClassifier(max_depth=depth)
                X_sampled = X_train.sample(frac=1, axis='rows', replace = True)
                y_sampled = y_train[X_sampled.index]
            else:
                base = tree.DecisionTreeRegressor(max_depth=depth)
                X_sampled = X_train.sample(frac=1, axis= 'rows', replace = True, random_state=1234)
                y_sampled = y_train[X_sampled.index]
            base.fit(X_sampled,y_sampled)
            y_hat = base.predict(X_test).reshape(1,-1)
            various_models[rounds]=y_hat
        if flag:
            main_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=0,arr=various_models)
            bias=np.sum(main_pred!=y_test)/y_test.size
            variance=np.zeros(y_hat.shape)
            for model in various_models:
                variance+=(model!=main_pred).astype(int)
            variance/=number_of_rounds
            variance=variance.sum()/y_test.shape[0]
        else:
            main_pred = np.mean(various_models,axis=0)
            bias = np.sum((main_pred-y_test)**2)/y_test.size
            variance = np.sum((main_pred-various_models)**2)/various_models.size
            #variance = np.var(y_hat)
        '''out.loc[depth, 'Bias^2'] = bias
        out.loc[depth, 'Variance'] = variance'''
        bias_plot.append(bias)
        variance_plot.append(variance)
        print(f"for depth {depth}")
        print(f"Bias: {bias}")
        print(f"Variance: {variance}")
       # print(out)
        #out=normalize(out)
            #bias_plot.append(bias/len(y_test))
            #variance_plot.append(variance/len(y_hat))
        #print(out)
    variance_plot = normalize(variance_plot)
    bias_plot = normalize(bias_plot)
    if flag:
        dataset = 'classifier'
    else:
        dataset = 'regressor'
    plt.plot([i for i in range(1,depths)],bias_plot,label='Bias')
    plt.plot([i for i in range(1,depths)],variance_plot,label='Variance')
    plt.title(f'Bias and Variance vs Depth/Complexity for {dataset}')
    plt.xlabel('Depth/Complexity')
    plt.legend()
    plt.show()
    return plt
depth = 16
print("Regression:")
X,y = Regression()
X_train,X_test,y_train,y_test = test_train(X,y)
fig1 =bias_variance(X_train,X_test,y_train,y_test,depth,number_of_rounds=200)
print("Classification:")
X,y = Classification()
X_train,X_test,y_train,y_test = test_train(X,y)
bias_variance(X_train,X_test,y_train,y_test,depth,number_of_rounds=200)
