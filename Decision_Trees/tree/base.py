"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)

@dataclass
# creating class for a decision tree node
class Node():
    def __init__(self):
        """
        Initialisisng attributes for tree node
        """
        self.children = {}
        self.Split_Value = None
        self.attr_id = None
        self.value = None
        self.isLeaf = False
        self.isAttrCategory = False

class DecisionTree():
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to
    
    def __init__(self, criterion, max_depth=None):
        """
        Initialising hyperparameters for Tree
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None


    def fit_data(self,X,y,currdepth):

        Current_Node = Node()   # Creating a new Tree Node

        attr_id = -1
        split_value = None
        optimal_measure = None
        """
        Splitting problem into two types:
            Output (category based) - Classification problem
            Outpur (Real or continous) - Regression problem
        """
        if(y.dtype.name=="category"):
            """
            Classification Problem
            """
            classes = np.unique(y)
            if(classes.size==1):
                Current_Node.isLeaf, Current_Node.isAttrCategory, Current_Node.value = True, True, classes[0]
                return Current_Node
            if(self.max_depth!=None):
                if(self.max_depth==currdepth):
                    Current_Node.isLeaf, Current_Node.isAttrCategory, Current_Node.value = True, True, y.value_counts().idxmax()
                    return Current_Node
            if(len(X.columns)==0):
                Current_Node.isLeaf, Current_Node.isAttrCategory, Current_Node.value = True, True, y.value_counts().idxmax()
                return Current_Node

            for i in X.columns:
                """
                Checking each type of attributes{catagorical or numerical}
                """
                X_attribute = X[i]

                if(X_attribute.dtype.name=="category"):
                    """
                    Discreate Input and Discreate Output
                    """
                    measure = None
                    if(self.criterion=="information_gain"):         # Criterion: Information Gain
                        measure = information_gain(y,X_attribute)
                    else:                                           # Criterion: Gini_index
                        classes_i = np.unique(X_attribute)
                        s = 0
                        for j in classes_i:
                            y_sub = pd.Series([y[k] for k in range(y.size) if X_attribute[k]==j])
                            s += y_sub.size*gini_index(y_sub)
                        measure = (s/X_attribute.size)
                    if(optimal_measure!=None):
                        if(optimal_measure<measure):
                            attr_id = i
                            optimal_measure = measure
                            split_value = None
                    else:
                        attr_id = i
                        optimal_measure = measure
                        split_value = None
                
                else:
                    """
                    Real Input and Discreate Output
                    """
                    X_attribute_sort = X_attribute.sort_values()
                    for j in range(X_attribute_sort.size-1):
                        curr_index = X_attribute_sort.index[j]
                        next_index = X_attribute_sort.index[j+1]
                        if(y[curr_index]!=y[next_index]):
                            measure = None
                            #print(type(X_attribute[curr_index]),type(X_attribute[next_index]))
                            Split_Value = (X_attribute[curr_index]+X_attribute[next_index])/2
                            
                            if(self.criterion=="information_gain"):                 # Criterion: Information Gain
                                temp_attr = pd.Series(X_attribute<=Split_Value)
                                measure = information_gain(y,temp_attr)
                            
                            else:                                                   # Criterion: Gini_index
                                y_sub1 = pd.Series([y[k] for k in range(y.size) if X_attribute[k]<=Split_Value])
                                y_sub2 = pd.Series([y[k] for k in range(y.size) if X_attribute[k]>Split_Value])
                                measure = y_sub1.size*gini_index(y_sub1) + y_sub2.size*gini_index(y_sub2)
                                measure =  (measure/y.size)
                            if(optimal_measure!=None):
                                if(optimal_measure<measure):
                                    attr_id, optimal_measure, split_value = i, measure, Split_Value
                            else:
                                attr_id, optimal_measure, split_value = i, measure, Split_Value
        
        # Regression Problems
        else:
            """
            Regression Problem
            """
            if(self.max_depth!=None):
                if(self.max_depth==currdepth):
                    Current_Node.isLeaf, Current_Node.value = True, y.mean()
                    return Current_Node
            if(y.size==1):
                Current_Node.isLeaf, Current_Node.value = True, y[0]
                return Current_Node
            if(len(X.columns)==0):
                Current_Node.isLeaf, Current_Node.value = True, y.mean()
                return Current_Node

            for i in X.columns:
                """
                Checking each type of attributes{catagorical or numerical}
                """
                X_attribute = X[i]
                if(X_attribute.dtype.name=="category"):
                    """
                    Discreate Input and Real Output
                    """
                    classes_i = np.unique(X_attribute)
                    measure = 0
                    for j in classes_i:
                        y_sub = pd.Series([y[k] for k in range(y.size) if X_attribute[k]==j])
                        measure += y_sub.size*np.var(y_sub)
                    if(optimal_measure!=None):
                        if(optimal_measure>measure):
                            optimal_measure, attr_id, split_value = measure, i, None
                    else:
                        optimal_measure, attr_id, split_value = measure, i, None
                
                else:
                    """
                    Real Input Real Output
                    """
                    X_attribute_sort = X_attribute.sort_values()
                    for j in range(y.size-1):
                        curr_index = X_attribute_sort.index[j]
                        next_index = X_attribute_sort.index[j+1]
                        Split_Value = (X_attribute[curr_index]+X_attribute[next_index])/2
                        y_sub1 = pd.Series([y[k] for k in range(y.size) if X_attribute[k]<=Split_Value])
                        y_sub2 = pd.Series([y[k] for k in range(y.size) if X_attribute[k]>Split_Value])
                        measure = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2)
                        if(optimal_measure!=None):
                            if(optimal_measure>measure):
                                attr_id, optimal_measure, split_value = i, measure, Split_Value
                        else:
                            attr_id, optimal_measure, split_value = i, measure, Split_Value
        # Checking the type of split
        if(split_value==None): # when current Node is category based
            Current_Node.isAttrCategory = True
            Current_Node.attr_id = attr_id
            classes = np.unique(X[attr_id])
            for j in classes:
                y_new = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]==j], dtype=y.dtype)
                X_new = X[X[attr_id]==j].reset_index().drop(['index',attr_id],axis=1)
                Current_Node.children[j] = self.fit_data(X_new, y_new, currdepth+1)
        # when current Node is split based
        else:
            Current_Node.attr_id = attr_id
            Current_Node.Split_Value = split_value
            """for k in range(y.size):
                print(k,y[k])"""
            y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]<=split_value], dtype=y.dtype)
            X_new1 = X[X[attr_id]<=split_value].reset_index().drop(['index'],axis=1)
            y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]>split_value], dtype=y.dtype)
            X_new2 = X[X[attr_id]>split_value].reset_index().drop(['index'],axis=1)
            Current_Node.children["lessThan"] = self.fit_data(X_new1, y_new1, currdepth+1)
            Current_Node.children["greaterThan"] = self.fit_data(X_new2, y_new2, currdepth+1)
        
        return Current_Node


    def fit(self, X, y):
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        self.head = self.fit_data(X,y,0)


    def predict(self, X):
        y_hat = []                  
        for i in range(X.shape[0]):
            X_Sample = X.iloc[i,:]          
            head = self.head
            while(not head.isLeaf):                            
                if(head.isAttrCategory):                       
                    #print(head.attr_id,X_Sample[head.attr_id])
                    head = head.children[X_Sample[head.attr_id]]
                else:                                      
                    #print(X_Sample[head.attr_id])
                    #print(head.Split_Value)
                    if(X_Sample[head.attr_id]<=head.Split_Value):
                        head = head.children["lessThan"]
                    else:
                        head = head.children["greaterThan"]
            
            y_hat.append(head.value)      #if leaf then decision arrived
        
        y_hat = pd.Series(y_hat)
        return y_hat


    def plotTree(self, root, depth):
        if(root.isLeaf):
            if(root.isAttrCategory):
                return "Class "+str(root.value)
            else:
                return "Value "+str(root.value)

        s = ""
        if(root.isAttrCategory):
            for i in root.children.keys():
                s += "?("+str(root.attr_id)+" == "+str(i)+")\n" 
                s += "\t"*(depth+1)
                s += str(self.plotTree(root.children[i], depth+1)).rstrip("\n") + "\n"
                s += "\t"*(depth)
            s = s.rstrip("\t")
        else:
            s += "?("+str(root.attr_id)+" <= "+str(root.Split_Value)+")\n"
            s += "\t"*(depth+1)
            s += "Y: " + str(self.plotTree(root.children["lessThan"], depth+1)).rstrip("\n") + "\n"
            s += "\t"*(depth+1)
            s += "N: " + str(self.plotTree(root.children["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
        return s
           

    def plot(self):
        head = self.head
        tree = self.plotTree(head,0)
        print(tree)
