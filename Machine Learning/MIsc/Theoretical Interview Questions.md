1. **What is supervised machine learning?**
	A case when we have both features (the matrix X) and the labels (the vector y)

2. Linear Regression
	- **What is regression? Which models can you use to solve a regression problem?**
		Regression is a part of supervised ML. Regression models investigate the relationship between a dependent (target) and independent variable (s) (predictor). Here are some common regression models
		- _Linear Regression_ establishes a linear relationship between target and predictor (s). It predicts a numeric value and has a shape of a straight line.
		- _Polynomial Regression_ has a regression equation with the power of independent variable more than 1. It is a curve that fits into the data points.
		- _Ridge Regression_ helps when predictors are highly correlated (multicollinearity problem). It penalizes the squares of regression coefficients but doesn’t allow the coefficients to reach zeros (uses L2 regularization).
		- _Lasso Regression_ penalizes the absolute values of regression coefficients and allows some of the coefficients to reach absolute zero (thereby allowing feature selection).
	
	- **What is linear regression? When do we use it?**
		Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y).
		With a simple equation:
			$y = B_0 + B_1*x_1 + ... + B_n * x_N$
		
		B is regression coefficients, x values are the independent (explanatory) variables and y is dependent variable.
		
		The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.
		
		Simple linear regression: 
			$y = B_0 + B_1*x_1$
		Multiple linear regression:
			$y = B_0 + B_1*x_1 + ... + B_n * x_N$

	- **What are the main assumptions of linear regression?**
		There are several assumptions of linear regression. If any of them is violated, model predictions and interpretation may be worthless or misleading.
		1.  **Linear relationship** between features and target variable.
		2. **Additivity** means that the effect of changes in one of the features on the target variable does not depend on values of other features. For example, a model for predicting revenue of a company have of two features - the number of items _a_sold and the number of items _b_ sold. When company sells more items _a_ the revenue increases and this is independent of the number of items _b_ sold. But, if customers who buy _a_ stop buying _b_, the additivity assumption is violated.
		3. Features are not correlated (no **collinearity**) since it can be difficult to separate out the individual effects of collinear features on the target variable.
		4. Errors are independently and identically normally distributed (yi = B0 + B1*x1i + ... + errori):
			- No correlation between errors (consecutive errors in the case of time series data).
			- Constant variance of errors - **homoscedasticity**. For example, in case of time series, seasonal patterns can increase errors in seasons with higher activity.
			- Errors are normally distributed, otherwise some features will have more influence on the target variable than to others. If the error distribution is significantly non-normal, confidence intervals may be too wide or too narrow.
	
	- **What’s the normal distribution? Why do we care about it?**
		The normal distribution is a continuous probability distribution whose probability density function takes the following formula: 
		- ![[Screenshot 2022-12-27 at 4.27.34 PM.png|300]]
		where μ is the mean and σ is the standard deviation of the distribution.
		
		The normal distribution derives its importance from the **Central Limit Theorem**, which states that if we draw a large enough number of samples, their mean will follow a normal distribution regardless of the initial distribution of the sample, i.e **the distribution of the mean of the samples is normal**. It is important that each sample is independent from the other. This is powerful because it helps us study processes whose population distribution is unknown to us.
	
	- **How do we check if a variable follows the normal distribution?**
		1.  Plot a histogram out of the sampled data. If you can fit the bell-shaped "normal" curve to the histogram, then the hypothesis that the underlying random variable follows the normal distribution can not be rejected.
		2. Check Skewness and Kurtosis of the sampled data. Skewness = 0 and kurtosis = 3 are typical for a normal distribution, so the farther away they are from these values, the more non-normal the distribution.
		3. Use Kolmogorov-Smirnov or/and Shapiro-Wilk tests for normality. They take into account both Skewness and Kurtosis simultaneously.
		4. Check for Quantile-Quantile plot. It is a scatterplot created by plotting two sets of quantiles against one another. Normal Q-Q plot place the data points in a roughly straight line.
	