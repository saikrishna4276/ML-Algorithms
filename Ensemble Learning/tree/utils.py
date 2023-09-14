
import pandas as pd
import numpy as np
def entropy(Y,sample_weight):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    unique_response={}
    #print(Y)
    total=np.sum(sample_weight)
    for j in range(Y.size):
        #print(j)
        if Y.iat[j] not in unique_response:    #creating dictionary
            unique_response[Y.iat[j]]=sample_weight.iat[j]        # of unique catagories in output variable
        else:
            unique_response[Y.iat[j]]+=sample_weight.iat[j]
    entropy=0
    for i in unique_response:
        probab=(unique_response[i]/total)
        entropy+=(-(probab)*np.log2(probab))
    return entropy

def gini_index(Y,sample_weight):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """

    unique_response={}
    total=np.sum(sample_weight)
    for j in range(Y.size):
        #print(j)
        if Y.iat[j] not in unique_response:    #creating dictionary
            unique_response[Y.iat[j]]=sample_weight.iat[j]       # of unique catagories in output variable
        else:
            unique_response[Y.iat[j]]+=sample_weight.iat[j]
    gini=0
    for i in unique_response:
        probab=(unique_response[i]/total)
        gini+=probab**2
    return gini


def information_gain(Y, attr, sample_weight):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    info_gain=entropy(Y,sample_weight)
    #print(info_gain)
    tot_size=np.sum(sample_weight)
    weighted_attr={}
    for i in range(attr.size):
        if attr.iat[i] not in weighted_attr:
            weighted_attr[attr.iat[i]]=[[Y.iat[i],sample_weight.iat[i]]]
        else:
            weighted_attr[attr.iat[i]].append([Y.iat[i],sample_weight.iat[i]])
    for i in weighted_attr:
        weighted_attr[i]=np.transpose(weighted_attr[i])
        #print(weighted_attr[i])
        #print(weighted_attr[i][0])
        #print(weighted_attr[i][1])
        info_gain-=(np.sum(weighted_attr[i][1])/tot_size)*entropy(pd.Series(weighted_attr[i][0]),pd.Series(weighted_attr[i][1]))
    return info_gain

