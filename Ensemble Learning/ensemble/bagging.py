from tree.base import WeightedDecisionTree
import pandas as pd
import numpy as np
import sklearn.tree
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore') 
from sklearn.utils.extmath import weighted_mode
from joblib import Parallel, delayed
import time
import threading
class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100,n_jobs='single'):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.models = []
        self.fitt = []
        self.n_jobs = n_jobs
        
    def fit_model(self,X,y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.type = y.dtype
        #for i in range(self.n_estimators):
        X_sampled = X.sample(frac=1, axis= 'rows', replace = True)
        y_sampled = y[X_sampled.index]
        X_sampled=X_sampled.reset_index(drop=True)
        y_sampled=y_sampled.reset_index(drop=True)
        model = self.base_estimator()
        model.fit(X_sampled, y_sampled)
        self.models.append(model)
        self.fitt.append([X_sampled, y_sampled])
        return self.models, self.fitt
        #print(self.models) 
    
    def fit(self,X,y):
        
        if self.n_jobs=='single':
            
            for i in range(self.n_estimators):
                X_sampled = X.sample(frac=1, axis= 'rows', replace = True)
                y_sampled = y[X_sampled.index]
                X_sampled=X_sampled.reset_index(drop=True)
                y_sampled=y_sampled.reset_index(drop=True)
                model = self.base_estimator()
                model.fit(X_sampled, y_sampled)
                self.models.append(model)
                self.fitt.append([X_sampled, y_sampled])
                
            
        else:
            #self.models, self.fitt = Parallel(n_jobs=self.n_jobs,verbose=100)(delayed(self.fit_model)(X,y) for i in range(self.n_estimators))
            threads = []
            for i in range(self.n_estimators):
                threads.append(threading.Thread(target=self.fit_model, args=(X, y,)))
            for i in range(self.n_estimators):
                threads[i].start()
            for i in range(self.n_estimators):
                threads[i].join()
        
        '''
        self.n_jobs=n_jobs
        start_time = time.perf_counter()
        result = Parallel(n_jobs=self.n_jobs)(delayed(self.fit_model)(X,y) for i in range(1,self.n_estimators))
        finish_time = time.perf_counter()
        return finish_time-start_time'''
          

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = np.zeros(len(X))
        for estimator in self.models:
            predictions += estimator.predict(X)
        return pd.Series(np.round(predictions / len(self.models)))
        prediction = None
        index = 1 
        for model in self.models:
            if prediction is None:
                prediction = pd.Series(model.predict(X)).to_frame()
            else:
                prediction[index] = model.predict(X)
                index+=1
        #print(self.models)
        #print(prediction)
        
        return prediction.mode(axis=1)[0]
        

    def plot(self, X, y):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        
        Zs = []
        fig1, ax1 = plt.subplots(
            1, len(self.models), figsize=(5*len(self.models), 4))

        x_min, x_max = X[0].min(), X[0].max()
        y_min, y_max = X[1].min(), X[1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, tree in enumerate(self.models):
            X_tree, y_tree = self.fitt[i]

            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Zs.append(Z)
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)
            colors = np.array(["red", "blue", "green"])
            for y_label in y.unique():
                idx = y_tree == y_label
                id = list(y_tree.cat.categories).index(y_tree[idx].iloc[0])
                ax1[i].scatter(X_tree.loc[idx, 0], X_tree.loc[idx, 1], c=colors[id],cmap=plt.cm.RdYlBu, edgecolor='black', s=30,label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()

        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Zs = np.array(Zs)
        com_surface, _ = weighted_mode(Zs, np.ones(Zs.shape))
        cs = ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X.loc[idx, 0], X.loc[idx, 1], c = colors[id], 
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend()
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)

        # Saving Figures
        
        return fig1, fig2
pass
