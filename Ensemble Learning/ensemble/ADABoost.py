import sklearn
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class AdaBoostClassifier():
    def __init__(self, base_estimator = DecisionTreeClassifier, n_estimators=10, classes =2, max_depth=1): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = [None]*self.n_estimators
        self.weights = [0]*self.n_estimators #weight given to each model say alpha
        self.mod_weights = [None]*self.n_estimators
        self.classes=classes
        self.max_depth=max_depth
    def fit(self, X: pd.DataFrame, y:pd.Series):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        #self.classes=np.unique(y)
        weight = np.array([i/y.size for i in range(1,y.size+1)])
        for estimator in range(self.n_estimators):
            clf = self.base_estimator(max_depth=self.max_depth,criterion='entropy')
            #print(weight)
            clf.fit(X,y,sample_weight=weight)
            pred = pd.Series(clf.predict(X))
            #print(pred)
            wrong_classified = pred!=y
            error = np.sum(weight[wrong_classified])/np.sum(weight)
            alpha = 0.5*np.log2((1-error)/error)+np.log(self.classes-1)
            #print(alpha)
            weight[wrong_classified] = weight[wrong_classified]*np.exp(alpha)
            weight[~wrong_classified] = weight[~wrong_classified]*np.exp(-alpha)
            weight=weight/np.sum(weight)
            #print(sklearn.tree.export_text(clf))
            self.models[estimator]=clf
            self.weights[estimator]=alpha
            self.mod_weights[estimator]=weight
        '''for i in self.models:
            print(sklearn.tree.export_text(i))'''
    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        prediction = []
        weight_class=[{j:0 for j in range(self.classes)} for i in range(X.shape[0])]
        for i in range(len(self.models)):
            y_hat = pd.Series(self.models[i].predict(X))
            for j in range(y_hat.size):
                weight_class[j][y_hat.iat[j]]+=self.weights[i]
        #print(weight_class)
        for i in weight_class:
            prediction.append(max(i, key = i.get)  )
            """if prediction is None:
                prediction = self.weights[i]*pd.Series(self.models[i].predict(X))
            else:
                prediction += self.weights[i]*pd.Series(self.models[i].predict(X))
        for i in range(len(prediction)):
            prediction[i]=-1 if prediction[i]<0 else 1"""
        return pd.Series(prediction)

    def plot(self,X,y):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        assert(len(list(X.columns)) == 2)
        color = ["r", "b", "g","y"]
        plot_step = 0.02
        plot_step_coarser = 0.5
        cmap = plt.cm.RdYlBu
        Zs = None
        fig1, ax1 = plt.subplots(1, len(self.models), figsize=(5*len(self.models), 4))
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        x_min, x_max = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
        y_min, y_max = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))
        #print(xx.shape)
        self.alpha = (self.weights-min(self.weights))/(max(self.weights)-min(self.weights))
        for i, (alpha_m, tree) in enumerate(zip(self.alpha, self.models)):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            print(sklearn.tree.export_text(tree))
            
            # _ = ax1.add_subplot(1, len(self.models), i + 1)
            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if Zs is None:
                Zs = alpha_m*Z
            else:
                Zs += alpha_m*Z
            cs = ax1[i].contourf(xx, yy, Z, cmap=cmap)
            cs1 = ax2.contourf(xx, yy, Z, alpha = alpha_m, cmap=cmap)
            xx_coarser, yy_coarser = np.meshgrid(np.arange(x_min, x_max, plot_step_coarser),np.arange(y_min, y_max, plot_step_coarser),)
            Z_points_coarser = tree.predict(np.c_[xx_coarser.ravel(), yy_coarser.ravel()]).reshape(xx_coarser.shape)
            ax1[i].scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)
            ax2.scatter(xx_coarser,yy_coarser,s=30,c=Z_points_coarser,cmap=cmap,edgecolors="black",)
        
            for y_label in range(self.classes):
                idx = y == y_label
                id = list(y.cat.categories).index(y[idx].iloc[0])
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],cmap=ListedColormap(["r", "g", "b"]), edgecolor='black', s=20,label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()
        for y_label in range(self.classes):
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], c=color[id],cmap=cmap, edgecolor='black', s=30,label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs1, ax=ax2, shrink=0.9)

        fig,ax=plt.subplots(1,self.n_estimators,figsize=(5*len(self.models), 4))
        for i in range(self.n_estimators):
            alpha,Dt,weights=self.weights[i],self.models[i],self.mod_weights[i]
            y_hat=list(Dt.predict(X))
            weight=list(weights/np.max(weights)*40)
            x_axis=list(X.iloc[:,0])
            y_axis=list(X.iloc[:,1])
            ax[i].scatter(x_axis,y_axis,s=weight)
            temp="Alpha: "+str(alpha)
            ax[i].set_title(temp)
        return fig1, fig2, fig
        pass
