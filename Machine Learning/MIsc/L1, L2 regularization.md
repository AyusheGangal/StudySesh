### <mark style="background: #D2B3FFA6;">Over-fitting</mark>
- Overfitting happens when model learns signal as well as noise in the training data and wouldn’t perform well on new data on which model wasn’t trained on.
- Complex models, like the Random Forest, Neural Networks, and XGBoost are more prone to overfitting. Simpler models, like linear regression, can overfit too – this typically happens when there are more features than the number of instances in the training data.
- Now, there are few ways you can avoid overfitting your model on training data like cross-validation sampling, reducing number of features, pruning, regularization etc.

### <mark style="background: #D2B3FFA6;">Regularization</mark>
- Regularization basically adds the penalty as model complexity increases.
- Regularization parameter (lambda) penalizes all the parameters except intercept so that model generalizes the data and won’t overfit.![[Screenshot 2023-01-27 at 12.27.31 PM.png|500]]
- So as the complexity is increasing, regularization will add the penalty for higher terms. This will decrease the importance given to higher terms and will bring the model towards less complex equation.

>[! Note]
>A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.
>_The key difference between these two is the penalty term_

##### **<mark style="background: #ABF7F7A6;">Lasso regression</mark>**
- Stands for "_Least Absolute Shrinkage and Selection Operator_".
- It adds “_absolute value of magnitude_” of coefficient as penalty term to the loss function.
- ![[Screenshot 2023-01-27 at 12.33.47 PM.png|300]]
- if _lambda_ is zero then we will get back OLS whereas very large value will make coefficients zero hence it will under-fit.

##### **<mark style="background: #ABF7F7A6;">Ridge regression</mark>**
- **Ridge regression** adds “_squared magnitude_” of coefficient as penalty term to the loss function.
- ![[Screenshot 2023-01-27 at 12.36.19 PM.png|300]]
- if _lambda_ is zero then we will get back OLS. However, if _lambda_ is very large then it will add too much weight and it will lead to under-fitting. Having said that it’s important how _lambda_ is chosen. This technique works very well to avoid over-fitting issue.

>[!Note]
>The **key difference** between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for **feature selection** in case we have a huge number of features.

Traditional methods like cross-validation, stepwise regression to handle overfitting and perform feature selection work well with a small set of features but these techniques are a great alternative when we are dealing with a large set of features.