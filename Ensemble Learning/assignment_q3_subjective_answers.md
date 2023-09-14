## Q3. a
Implemented ADABoost in ensemble/ADABoost.py
## Q3 .b
Adaboost has been implemented with default `max_depth` to be 1 (`decision stump`). Adaboost has been implemneted for both multiclass and binary class.

Iris dataset has been taken for multi class representation and plotted for two features i.e. `sepal lenght` and `sepal width`

The plots obtained for multi class are as follows:

Alpha Values:

![Alt text](./multi%20class%20adaboost%20for%20alphas%20sepal%20length%20(cm)%20sepal%20width%20(cm).png "multi class adaboost for alphas sepal length (cm) sepal width (cm).png")

Individual Estimators:

![Alt text](./multi%20class%20adaboost%20for%20feature%20sepal%20length%20(cm)%20sepal%20width%20(cm).png) "Individual Estimators"

Combined Decision Surface:

![Alt text](./multi%20class%20common%20adaboost%20for%20feature%20sepal%20length%20(cm)%20sepal%20width%20(cm).png) "Common surface"

`make_classification()` from sklearn is used to generate data for binary class and the plots are as below:

Alpha Values:

![Alt text](./Alphas%20for%20adaboost.png "Alpha")

Individual Estimators:

![Alt text](./Adaboost%20for%20individual%20estimator.png "Individual")

Combined Decision Surface:

![Alt text](./Adaboost%20for%20combined.png "Combined")

Accuracy comparision for implemented and sklearn model is:
```
Accuracy of implemented adaboost model for: 0.99
Accuracy of sklearn model: 0.97
```