import numpy as np
import pandas as pd


def entropy(Y):
    unique_response={}
    total=Y.size
    for j in range(Y.size):
        #print(j)
        if Y.iat[j] not in unique_response:    #creating dictionary
            unique_response[Y.iat[j]]=1        # of unique catagories in output variable
        else:
            unique_response[Y.iat[j]]+=1
    entropy=0
    for i in unique_response:
        probab=(unique_response[i]/total)
        entropy+=(-(probab)*np.log2(probab))
    return entropy



def gini_index(Y):
    unique_response={}
    total=Y.size
    for j in range(Y.size):
        #print(j)
        if Y.iat[j] not in unique_response:    #creating dictionary
            unique_response[Y.iat[j]]=1        # of unique catagories in output variable
        else:
            unique_response[Y.iat[j]]+=1
    gini=0
    for i in unique_response:
        probab=(unique_response[i]/total)
        gini+=probab**2
    return gini

    


def information_gain(Y, attr):
    info_gain=entropy(Y)
    #print(info_gain)
    tot_size=Y.size
    weighted_attr={}
    for i in range(attr.size):
        if attr.iat[i] not in weighted_attr:
            weighted_attr[attr.iat[i]]=[Y.iat[i]]
        else:
            weighted_attr[attr.iat[i]].append(Y.iat[i])
    for i in weighted_attr:
        info_gain-=(len(weighted_attr[i])/tot_size)*entropy(pd.Series(weighted_attr[i]))
    return info_gain

