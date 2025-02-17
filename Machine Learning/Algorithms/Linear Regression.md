**Linear Regression** is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data. It's widely employed in predictive analytics and machine learning for tasks involving continuous outcomes. It is a **supervised learning** technique used for **predicting continuous values** based on input features.

Linear Regression models the relationship between **input variables (features)** and **output (target variable)** by fitting a **straight line** to the data.

In **Simple Linear Regression** (one feature), the relationship is modeled as:
$$y=wx+b$$
where:
- $y$ = predicted output (dependent variable). It is also called the _target variable_ in machine learning, or the _dependent variable_ in statistical modeling. It represents the continuous value that we are trying to predict.
- $x$ = input feature (independent variable). In machine learning, x is referred to as the _feature,_ while in statistics, it is called the _independent variable_. It represents the information given to us at any given time.
- $w$ = weight (slope of the line), is the _regression coefficient_ or scale factor. In classical statistics, it is the equivalent of the slope on the best-fit straight line that is produced after the linear regression model has been fitted.
- $b$ = bias (intercept). 

The goal of the regression analysis (modeling) is to find the values for the unknown parameters of the equation; that is, to find the values for the weights w.

For **Multiple Linear Regression** (multiple features):
$$\hat{y}=w_1x_1+w_2x_2+⋯+w_nx_n+b$$

or in **matrix form**:
$\hat{Y}=XW+b$

where:
- $X$ = **feature matrix** (shape: m×nm \times nm×n)
- $W$ = **weight vector** (shape: n×1n \times 1n×1)
- $b$ = **bias (intercept)** (scalar)

Despite their differences, both the simple and multiple regression models are linear models — they adopt the form of a _linear_ equation. This is called the **linear assumption**. Quite simply, it means that we assume that the type of relationship between the set of independent variables and independent features is linear.
### **Assumptions of Linear Regression**
Linear Regression relies on several key assumptions:
1. **Linearity** – The relationship between the independent and dependent variable is **linear**.
2. **Independence** – Observations are independent of each other.
3. **Homoscedasticity** – The variance of errors remains **constant** across all values of $x$.
4. **No Multicollinearity** – Features should not be **highly correlated** with each other.
5. **Normality of Residuals** – Errors (residuals) should be **normally distributed**.

Violating these assumptions can lead to **poor model performance** or **misinterpretation** of results.

### Training an LR Model
We train the linear regression algorithm with a method named **_Ordinary Least Squares_** — **_OLS_**(or just Least Squares). The goal of training is to find the weights $w_i$ in the linear equation $y = w_0 + w_1x$

1. Random weight initialization. In practice, $w_0$ and $w_1$ are unknown at the beginning. The goal of the procedure is to find the appropriate values for these model parameters. To start the process, we set the values of the weights at random or just initialize with 0.

> There are other mathematical justifications such as **Xavier Initialization**, etc.

2. Input the initialized weights into the linear equation and generate a prediction for each observation point.

3. Calculate the **Residual Sum of Squares** _(RSS). Residuals, or error terms_, are the difference between each **_actual output_** and the **_predicted output_**.

> They are a point-by-point estimate of how well our regression function predicts outputs in comparison to true values. We obtain residuals by calculating _actual values — predicted values_ for each observation.

We **_square the residuals_** for each observation point and sum the residuals to reach our RSS.

The basis here is that a **_lower RSS_** means that our line of **_best fit_** comes closer to each data point and the vice versa.

4. Model parameter selection to minimize RSS. Machine learning approaches find the best parameters for the linear model by defining a _cost function_ and _minimizing_ it via **gradient descent.**

### **How Does Linear Regression Work?**

**Step 1: Hypothesis Function**
Linear Regression assumes that the output yyy can be written as a **linear combination** of input features:
$$\hat{y}=w_1x_1+w_2x_2+⋯+w_nx_n+b$$

The goal is to find the **best values for $w$ and $b$** that minimize the error.


**Step 2: Loss Function (Mean Squared Error - MSE)**
The error between predictions and actual values is measured using the **Mean Squared Error (MSE)**:$$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$where:
- $n$ = number of training samples
- $y_i$ = actual output
- $\hat{y_i}$= predicted output

✅ **MSE penalizes larger errors more**, making it sensitive to outliers.

**Step 3: Optimization Using Gradient Descent**
To find the optimal values of w and b, we use **Gradient Descent**, an iterative optimization algorithm.
#### **1. Compute Gradients (Partial Derivatives of MSE)**
**Gradient w.r.t $w$:**
$$\frac{\partial MSE}{\partial w} = -\frac{2}{m} \sum_{i=1}^{m} x_i (y_i - \hat{y}_i)$$

**Gradient w.r.t $b$:**
$$\frac{\partial MSE}{\partial b} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)
$$

#### **2. Update Parameters Using Gradient Descent**
$$w:=w−α\frac{∂MSE}{∂w}$$​where $\alpha$ = **learning rate** (controls step size).

🔹 **Too high α\alphaα** → Algorithm diverges.  
🔹 **Too low α\alphaα** → Convergence is slow.

**Step 4: Normal Equation (Closed-Form Solution - Optional)**
Instead of using gradient descent, we can directly compute the optimal weights using **Linear Algebra**:
$$W=(X^TX)^{−1}X^TY$$
✅ **Advantage:** No need to choose a learning rate.  
🚫 **Disadvantage:** Computationally expensive for large datasets ($O(n^3)$ complexity).

### **Overfitting & Regularization**

**Overfitting in Linear Regression**
- If the model learns **too much noise**, it **overfits** the training data and performs poorly on test data.
- A sign of overfitting is a **high training accuracy but low test accuracy**.

**Regularization: Ridge & Lasso Regression**
To prevent overfitting, we add a **penalty term** to the loss function:

🔹 **Ridge Regression (L2 Regularization)**
$$MSE+ \lambda \sum w^2$$
- Penalizes **large weights** and prevents overfitting.

🔹 **Lasso Regression (L1 Regularization)**
$$MSE+ \lambda \sum |w|$$
- Encourages **sparsity** (sets some weights to zero, selecting only important features).

✅ **Ridge → Shrinks weights but keeps all features.**  
✅ **Lasso → Selects the most important features.**
