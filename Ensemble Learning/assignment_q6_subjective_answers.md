Gradient Bossting is implmented using Decision Tree from sklearn and is tested using `q6_GradientBoosted.py`.

Number of estimators are varied and compared against sklearn.ensemble.GradientBoost and the results are as follows:

```
No. of estimators: 100
MSE of Gradient boosting implemented is 49.52603231749845
MSE of Gradient boosting from sklearn is 49.52603231749844
No. of estimators: 200
MSE of Gradient boosting implemented is 22.075936550244204
MSE of Gradient boosting from sklearn is 22.075936550244215
No. of estimators: 300
MSE of Gradient boosting implemented is 8.36162400734036
MSE of Gradient boosting from sklearn is 8.361624007340362
No. of estimators: 400
MSE of Gradient boosting implemented is 3.50788287272796
MSE of Gradient boosting from sklearn is 3.5078828727279587
No. of estimators: 500
MSE of Gradient boosting implemented is 1.4862970043134984
MSE of Gradient boosting from sklearn is 1.4862970043135015
No. of estimators: 600
MSE of Gradient boosting implemented is 0.6627131854893828
MSE of Gradient boosting from sklearn is 0.6627131854893841
No. of estimators: 700
MSE of Gradient boosting implemented is 0.3266764144099221
MSE of Gradient boosting from sklearn is 0.3266764144099223
No. of estimators: 800
MSE of Gradient boosting implemented is 0.144743764684239
MSE of Gradient boosting from sklearn is 0.1447437646842396
No. of estimators: 900
MSE of Gradient boosting implemented is 0.06651600125412466
MSE of Gradient boosting from sklearn is 0.06651600125412428
No. of estimators: 1000
MSE of Gradient boosting implemented is 0.03130931547015837
MSE of Gradient boosting from sklearn is 0.03130931547015823
```

The implementation is correct as MSE of sklearn is exactly same as implemented one. We can also observe that at estimators increase MSE is decreasing which is good upto certain rate, because for higher values of estimators, it overfits.