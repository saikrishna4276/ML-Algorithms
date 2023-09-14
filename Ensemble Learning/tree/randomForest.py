from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, max_attr ='sqrt'):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.Forest=[None]*n_estimators
        self.criteria=criterion
        self.max_attr = max_attr

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for i in range(self.n_estimators):
            X_sampled = X.sample(frac=0.75, axis= 'rows', replace = True)
            y_sampled = y[X_sampled.index]
            X_sampled=X_sampled.reset_index(drop=True)
            y_sampled=y_sampled.reset_index(drop=True)
            Dt=DecisionTreeClassifier(max_features=self.max_attr,criterion=self.criteria)
            Dt.fit(X_sampled,y_sampled)
            self.Forest[i]=Dt

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            Dt=self.Forest[i]
            res[:,i]=np.array(Dt.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a=list(res[i])
            y_hat[i]=max(set(a),key=a.count)
        return pd.Series(y_hat)

    def plot(self,X,y,plot=False,pair=None):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for i in range(self.n_estimators):
            plt.clf()
            tree.plot_tree(self.Forest[i])
            temp="Tree number"+str(i)
            plt.title(temp)
            plt.savefig(fname=f"RF_Classifier_fig{i+1}")
        if plot==True:
            [fig1,fig2]=self.dec_boundary(X,y,pair)
            return fig1, fig2
    def dec_boundary(self,X,y,pair=None):
        color = ["r", "y", "b"]
        plot_step = 0.02
        plot_step_coarser = 0.5
        cmap = plt.cm.RdYlBu
        Zs = None
        fig1, ax1 = plt.subplots(
            1, len(self.Forest), figsize=(5*len(self.Forest), 4))
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        x_min, x_max = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        y_min, y_max = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
        #features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

        #print(self.Forest)
        i=0
        
        for  tree in self.Forest:
            print("-----------------------------")
            #print("Tree Number: {}".format(i+1))
            print(sklearn.tree.export_text(tree))
            print("-----------------------------")
            
            
            # _ = ax1.add_subplot(1, len(self.Forest), i + 1)
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            cs = ax1[i].contourf(xx, yy, Z, cmap=cmap)
            #cs1 = ax2.contourf(xx, yy, Z, alpha = alpha_m, cmap=cmap)
            xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),np.arange(y_min, y_max, plot_step_coarser),)
            Z_points_coarser = tree.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
            ax1[i].scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
            fig1.colorbar(cs, shrink=0.9)
            ax2.scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
        
            for y_label in np.unique(y):
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],cmap=ListedColormap(["r", "y", "b"]), edgecolor='black', s=20,label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
            i+=1
        fig1.tight_layout()
        #fig1.show()
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
        fig2.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        cs = ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        cs1 = ax2.contourf(xx, yy, Z, cmap=cmap)
        xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),np.arange(y_min, y_max, plot_step_coarser),)
        Z_points_coarser = tree.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
        ax2.scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
            
        y_hat=list(self.predict(X))
        x_axis=list(X.iloc[:,0])
        y_axis=list(X.iloc[:,1])
        for y_label in np.unique(y):
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],cmap=cmap, edgecolor='black', s=30,label="Class: "+str(y_label))
        ax2.legend(loc="lower right")
        ax2.set_title("Random Forest")
        fig2.colorbar(cs1, shrink=0.9)   
        #fig1.savefig(fname='RF_fig1'+str(0)+str(1)+'.png')
        #fig2.savefig(fname='RF_fig2'+str(0)+str(1)+'.png')
        return fig1, fig2




class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.Forest=[None]*n_estimators
        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        X_temp1=X.copy()
        X_temp1["res"]=y
        for i in range(self.n_estimators):
            X_temp=X_temp1.sample(frac=0.6)
            Dt=DecisionTreeRegressor(max_features=2)
            Dt.fit(X_temp.iloc[:,:-1],X_temp.iloc[:,-1])
            self.Forest[i]=Dt
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        res=np.zeros((X.shape[0],self.n_estimators))
        for i in range(self.n_estimators):
            Dt=self.Forest[i]
            res[:,i]=np.array(Dt.predict(X))
        y_hat=np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_hat[i]=np.mean(res[i])
        return pd.Series(y_hat)
        
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for i in range(self.n_estimators):
            tree.plot_tree(self.Forest[i])
            temp="Tree number"+str(i)
            plt.title(temp)
            plt.savefig(fname=f"RF_Regressor_fig{i+1}")
            #plt.show()
        pass
