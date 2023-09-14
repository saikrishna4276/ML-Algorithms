
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    c=0
    for i in range(y.size):
        if y_hat.iloc[i]==y.iloc[i]:
            c+=1
    return c/y.size

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert y_hat.size == y.size
    c=0
    for i in range(y.size):
        if y.iloc[i]==y_hat.iloc[i] and y.iloc[i]==cls:
            c+=1
    return c/y_hat.value_counts()[cls]

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    c=0
    for i in range(y.size):
        if y.iloc[i]==y_hat.iloc[i] and y.iloc[i]==cls:
            c+=1
    return c/y.value_counts()[cls]
    

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert y_hat.size == y.size
    rms=0
    for i in range(y.size):
        rms+=(y_hat.iloc[i]-y.iloc[i])**2
    return (rms/y.size)**0.5

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert y_hat.size == y.size
    m_ae=0
    for i in range(y.size):
        m_ae+=abs(y_hat.iloc[i]-y.iloc[i])
    return m_ae/y.size

    