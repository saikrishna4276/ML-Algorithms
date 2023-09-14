## Q4

The plot of norm(theta) vs degree is as follows:

![Plot for degree vs theta](Plots/Question4/Norm(θ)_vs_Degree.png)



The norm of the coefficient vector represents the magnitude or length of the vector, and it can increase or decrease depending on the direction and magnitude of the updates to the coefficients.

If the updates to the coefficients are consistently in the same direction, then the norm of the coefficient vector will increase. This can happen when the learning rate, which controls the size of the updates, is too large. In this case, the algorithm may overshoot the minimum of the error function and keep moving in the same direction, causing the norm of the coefficient vector to increase.

On the other hand, if the updates to the coefficients are fluctuating in direction, then the norm of the coefficient vector will decrease. This can happen when the learning rate is too small, or when the algorithm is close to the minimum of the error function and the updates are becoming smaller and more erratic.

We can conclude that the as degree increases the norm of theta increases. This is due to overfitting, which is the reason why introduce penalty such as ```l1``` and ```l2``` to reduce this coefficients and thus reduce variance.

References:
[SKlearn](https://github.com/scikit-learn/scikit-learn/tree/main/sklearn/preprocessing)