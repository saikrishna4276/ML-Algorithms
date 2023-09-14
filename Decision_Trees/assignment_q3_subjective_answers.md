The data set for auto-mpg.data is downloaded from [automotive efficiency](https://archive.ics.uci.edu/ml/datasets/auto+mpg)

The data contained 9 attrubutes namely:
1. mpg: continuous
2. cylinders: multi-valued discrete
3. displacement: continuous
4. horsepower: continuous
5. weight: continuous
6. acceleration: continuous
7. model year: multi-valued discrete
8. origin: multi-valued discrete
9. car name: string (unique for each instance)
## Clean Data
Dropped car name as it is unique for each instance, few of the samples of ```horsepower``` is ```"?"```, so dropped those samples (6) (we can also consider those samples by replacing ```"?"``` with ```mean()```)

```horsepower``` is of type string so converted it to numeric and dtype of ```cylinders``` and ```origin``` is changed to ```"category"``` as they are discrete.

Here, the output or reesponse variable is ```mpg``` and the remaining are attributes or features.
## Train and Test using my model (Q1)
Once th data clean is done, trained 70% and tested on 30%, the results are as follows:
### My Model
```
Train Scores:
    RMSE:  1.6830127025750345
	MAE:  1.2881667048695677
Test Scores:
	RMSE:  6.571670343001028
	MAE:  5.222912455478249
```
### SKlearn Model
```
Test Scores:
	RMSE:  6.993783839503437
	MAE:  5.517978384957609
```
As per the results obtained, the test scores of my model and sklearn model is almost same  with RMSE being 6.57 and 6.99 respectively.