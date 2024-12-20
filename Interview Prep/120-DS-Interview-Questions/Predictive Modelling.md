(Given a Dataset) 
#### 1. <mark style="background: #ADCCFFA6;">Analyze this dataset and give me a model that can predict this response variable.</mark>
- Problem Determination -> Data Cleaning -> Feature Engineering -> Modeling
- Benchmark Models
    -  Linear Regression (Ridge or Lasso) for regression
    - Logistic Regression for Classification
- Advanced Models
    - Random Forest, Boosting Trees, and so on
        - Scikit-Learn, XGBoost, LightGBM, CatBoost
- Determine if the problem is classification or regression
- Plot and visualize the data.
- Start by fitting a simple model (multivariate regression, logistic regression), do some feature engineering accordingly, and then try some complicated models. Always split the dataset into train, validation, test dataset and use cross validation to check their performance.
- Favor simple models that run quickly and you can easily explain.
- Mention cross validation as a means to evaluate the model.

#### 2. <mark style="background: #D2B3FFA6;">What could be some issues if the distribution of the test data is significantly different than the distribution of the training data?</mark>
- The model that has high training accuracy might have low test accuracy. Without further knowledge, it is hard to know which dataset represents the population data and thus the generalizability of the algorithm is hard to measure. This should be mitigated by repeated splitting of train vs. test dataset (as in cross validation).
- When there is a change in data distribution, this is called the dataset shift. If the train and test data has a different distribution, then the classifier would likely overfit to the train data.
- This issue can be overcome by using a more general learning method.
- This can occur when:
    - P(y|x) are the same but P(x) are different. (covariate shift)
    -  P(y|x) are different. (concept shift)
- The causes can be:
    - Training samples are obtained in a biased way. (sample selection bias)
    - rain is different from test because of temporal, spatial changes. (non-stationary environments)
- Solution to covariate shift
    -  <mark style="background: #ABF7F7A6;">importance weighted cv</mark>

#### 3. <mark style="background: #FFF3A3A6;">What are some ways I can make my model more robust to outliers?</mark>
- We can have regularization such as L1 or L2 to reduce variance (increase bias).
- Changes to the algorithm:
    - Use tree-based methods instead of regression methods as they are more resistant to outliers. For statistical tests, use non parametric tests instead of parametric ones.
    - Use robust error metrics such as MAE or Huber Loss instead of MSE.
- Changes to the data:
    - [[Winsorizing the data]]
    - Transforming the data (e.g. log)
    - Remove them only if you’re certain they’re anomalies not worth predicting

>[!Note]
> Robustness can be defined as the capacity of a system or a model to remain stable and have only small changes (or none at all) when exposed to noise, or exaggerated inputs.
> 
> So a robust system or metric must be less affected by outliers. In this scenario it is easy to conclude that MSE may be less robust than MAE, since the squaring of the errors will enforce a higher importance on outliers. (MSE is more sensitive to outliers than MAE)
>


#### 4. <mark style="background: #FFB86CA6;">What are some differences you would expect in a model that minimizes squared error, versus a model that minimizes absolute error? In which cases would each error metric be appropriate?
</mark>
- MSE is more strict to having outliers. MAE is more robust in that sense, but is harder to fit the model for because it cannot be numerically optimized. So when there are less variability in the model and the model is computationally easy to fit, we should use MAE, and if that’s not the case, we should use MSE.
- MSE: easier to compute the gradient, MAE: linear programming needed to compute the gradient
- MAE more robust to outliers. If the consequences of large errors are great, use MSE
- MSE corresponds to maximizing likelihood of Gaussian random variables.

#### 5. <mark style="background: #D2B3FFA6;">What error metric would you use to evaluate how good a binary classifier is? What if the classes are imbalanced? What if there are more than 2 groups?</mark>
- Accuracy: proportion of instances you predict correctly. 
	- Pros: intuitive, easy to explain
	- Cons: works poorly when the class labels are imbalanced and the signal from the data is weak
- [[Area Under the Curve- Receiver Operating Characteristics (AUC-ROC)]]: plot fpr (false positive rate) on the x axis and tpr (true positive rate) on the y axis for different threshold. Given a random positive instance and a random negative instance, the AUC is the probability that you can identify who's who. 
	- Pros: Works well when testing the ability of distinguishing the two classes,
	- <mark style="background: #FFF3A3A6;">Cons: can’t interpret predictions as probabilities (because AUC is determined by rankings), so can’t explain the uncertainty of the model</mark>
- logloss/deviance: 
	- Pros: error metric based on probabilities
	- Cons: very sensitive to false positives, negatives

When there are more than 2 groups, we can have k binary classifications and add them up for logloss. Some metrics like AUC is only applicable in the binary case.

#### 6. <mark style="background: #ABF7F7A6;">What are various ways to predict a binary response variable? Can you compare two of them and tell me when one would be more appropriate? What’s the difference between these? (SVM, Logistic Regression, Naive Bayes, Decision Tree, etc.)</mark>
- Things to look at: N, P, linearly separable, features independent, likely to overfit, speed, performance, memory usage and so on.
- <mark style="background: #ADCCFFA6;">Logistic Regression</mark>
    - features roughly linear, problem roughly linearly separable
    - robust to noise, use l1,l2 regularization for model selection, avoid overfitting
    - the output come as probabilities
    - efficient and the computation can be distributed
    - can be used as a baseline for other algorithms
    - <mark style="background: #FF5582A6;">(-) can hardly handle categorical features</mark>
    
- <mark style="background: #ADCCFFA6;">SVM</mark>
    - with a nonlinear kernel, can deal with problems that are not linearly separable
    - <mark style="background: #FF5582A6;">(-) slow to train, for most industry scale applications, not really efficient</mark>
    
- <mark style="background: #ADCCFFA6;">Naive Bayes</mark>
    - computationally efficient when P is large by alleviating the curse of dimensionality
    - works surprisingly well for some cases even if the condition doesn’t hold
    - with word frequencies as features, the independence assumption can be seen reasonable. So the algorithm can be used in text categorization
    - <mark style="background: #FF5582A6;">(-) conditional independence of every other feature should be met</mark>
    
- <mark style="background: #ADCCFFA6;">Tree Ensembles</mark>
    - good for large N and large P, can deal with categorical features very well
    - non parametric, so no need to worry about outliers
    - GBT’s work better but the parameters are harder to tune
    - RF works out of the box, but usually performs worse than GBT
    
- <mark style="background: #ADCCFFA6;">Deep Learning</mark>
    - works well for some classification tasks (e.g. image)
    - used to squeeze something out of the problem

#### 7. <mark style="background: #FFB8EBA6;">What is regularization and where might it be helpful? What is an example of using regularization in a model?</mark>
- Regularization is useful for reducing variance in the model, meaning avoiding overfitting. That is, **Regularization attempts to reduce the variance of the estimator** by simplifying it, something that will increase the bias, in such a way that the expected error decreases.
- For example, we can use L1 regularization in Lasso regression to penalize large coefficients and automatically select features, or we can also use L2 regularization for Ridge regression to penalize the feature coefficients.

#### 8. <mark style="background: #BBFABBA6;">Why might it be preferable to include fewer predictors over many?</mark>
-   When we add irrelevant features, it increases model's tendency to overfit because those features introduce more noise. When two variables are correlated, they might be harder to interpret in case of regression, etc.
-   curse of dimensionality
-   adding random noise makes the model more complicated but useless
-   computational cost