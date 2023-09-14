The dataset for generation of weighted samples is as follows:

```python:
N = 200
P = 3
NUM_OP_CLASSES = 2
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=NUM_OP_CLASSES, class_sep=0.5)
X = pd.DataFrame(X)
y = pd.Series(y,dtype='category')
```
Weights are assigned to each target as follows:
```
weights=pd.Series(np.random.uniform(0,1,size=y.size))
```
Data has been shuffled using `sample(frac=1)` which shuffles data.

Default value of weight is 1 else the weight is given as per user.

The comparision results of sklearn and Implemented Weighted decision trees for different depths are as follows:

```
Depth: 1
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 2
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 3
Sklearn Accuracy: 0.8666666666666667
Implemented Weighted DecisionTree Accuracy: 0.8666666666666667
Depth: 4
Sklearn Accuracy: 0.8
Implemented Weighted DecisionTree Accuracy: 0.8
Depth: 5
Sklearn Accuracy: 0.7333333333333333
Implemented Weighted DecisionTree Accuracy: 0.8
```

So, it can be observed from above results that the accuracy of implemented weighted tree is in par with sklearn.

Plots:
```
Depth 1:

?(0 <= 0.004045013051359819)
	Y: Class 0
	N: Class 1

Depth 2:

?(0 <= 0.004045013051359819)
	Y: ?(1 <= 0.5382004610852587)
		Y: Class 0
		N: Class 0
	N: Class 1

Depth 3:

?(0 <= 0.004045013051359819)
	Y: ?(1 <= 0.5382004610852587)
		Y: ?(0 <= -0.9368513908927181)
			Y: Class 0
			N: Class 0
		N: ?(0 <= -0.5451447395508547)
			Y: Class 1
			N: Class 0
	N: Class 1

Depth 4:

?(0 <= 0.004045013051359819)
	Y: ?(1 <= 0.5382004610852587)
		Y: ?(0 <= -0.9368513908927181)
			Y: ?(0 <= -0.9827833749650142)
				Y: Class 0
				N: Class 1
			N: ?(0 <= -0.2588245906474876)
				Y: Class 0
				N: Class 0
		N: ?(0 <= -0.5451447395508547)
			Y: ?(0 <= -1.0327391295688262)
				Y: Class 0
				N: Class 1
			N: ?(0 <= -0.021695585311992477)
				Y: Class 0
				N: Class 0
	N: Class 1

Depth 5:

?(0 <= 0.004045013051359819)
	Y: ?(1 <= 0.5382004610852587)
		Y: ?(0 <= -0.9368513908927181)
			Y: ?(0 <= -0.9827833749650142)
				Y: Class 0
				N: Class 1
			N: ?(0 <= -0.2588245906474876)
				Y: Class 0
				N: ?(0 <= -0.22665714591763442)
					Y: Class 1
					N: Class 0
		N: ?(0 <= -0.5451447395508547)
			Y: ?(0 <= -1.0327391295688262)
				Y: Class 0
				N: ?(0 <= -0.5803487742844701)
					Y: Class 1
					N: Class 1
			N: ?(0 <= -0.021695585311992477)
				Y: Class 0
				N: ?(0 <= -0.009944371488954856)
					Y: Class 1
					N: Class 0
	N: Class 1
```