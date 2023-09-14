from typing import Union
import pandas as pd
import math
def accuracy(y_hat, y):
    assert(y_hat.size == y.size)
    # TODO: Write here
    c=0
    for i in range(y.size):
        if y_hat.iloc[i]==y.iloc[i]:
            c+=1
    return c/y.size

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    c=0
    for i in range(y.size):
        if y.iloc[i]==y_hat.iloc[i] and y.iloc[i]==cls:
            c+=1
    return c/y_hat.value_counts()[cls]
        

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    c=0
    for i in range(y.size):
        if y.iloc[i]==y_hat.iloc[i] and y.iloc[i]==cls:
            c+=1
    return c/y.value_counts()[cls]

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    rms=0
    for i in range(y.size):
        rms+=math.pow(y_hat.iloc[i]-y.iloc[i],2)
    return math.sqrt(rms/y.size)

    

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    m_ae=0
    for i in range(y.size):
        m_ae+=abs(y_hat.iloc[i]-y.iloc[i])
    return m_ae/y.size
