FakeData is created for N samples and M (Binary) Real Input (Discrete Input) features.

The time comparisions are done for all the four cases i.e. \
```Real Input and Real Output```\
```Real Input and Discrete Output```\
```Discrete Input and Real Output```\
```Discrete Input and Discrete Output```

The time comparisions are computed for all the cases of N samples varying from 45 to 54 and M features varying from 2 to 9 and are plotted using ```pandas.pivot``` and ```sns.heatmap``` for better undestanding.
The theorotical complexities of:\
## Real output: 
```
Theoretical Fit Time Complexity: O(MN^2) for creating each tree node, so total time taken is O(2^d*(MN^2))
Theoretical Predict Time Complexity: O(d) for creating each tree node
```
## Discrete output: 
```
Theoretical Fit Time Complexity: O(MN) for creating each tree node, so total time taken is O(c^d*(MN))
Theoretical Predict Time Complexity: O(d) for creating each tree node
```

The results obtained practically are plotted using seaborn.

The plots are as follows:

![Alt text](./d_d%20train.png "Discrete Input and Discrete Output")\
```Discrete Input and Discrete Output train```

![Alt text](./d_d%20test.png "Discrete Input and Discrete Output")\
```Discrete Input and Discrete Output test```

![Alt text](./r_d%20train.png "Discrete Input and Discrete Output")\
```     Real Input and Discrete Output train```

![Alt text](./r_d%20train.png "Discrete Input and Discrete Output")\
```     Real Input and Discrete Output test```

![Alt text](./d_r%20train.png "Discrete Input and Discrete Output")\
```     Discrete Input and Real Output train```

![Alt text](./d_r%20train.png "Discrete Input and Discrete Output")\
```     Discrete Input and Real Output test```

![Alt text](./r_r%20train.png "Discrete Input and Discrete Output")\
```     Real Input and Real Output train```

![Alt text](./r_r%20train.png "Discrete Input and Discrete Output")\
```     Real Input and Real Output test```

The time complexity for Real Input and Discrete output is large as compared to other cases.

The complexities are as follows for overall N samples and M features:\
```Real Input and Discrete Output > Real Input and Real Output test > Discrete Input and Real Output > Discrete Input and Discrete Output```
