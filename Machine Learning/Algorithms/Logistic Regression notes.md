
**1. Core Idea: Probability of a Binary Outcome**
* **What it is:** Logistic Regression is a classification algorithm used to predict the probability of a categorical dependent variable (target) with only two possible outcomes (binary). It's often used when the outcome is a Yes/No, True/False, 0/1, or Success/Failure. Despite the name "regression," it's a *classification* algorithm.
* **Goal:** To model the probability that a given input belongs to a specific category. The output is a probability value between 0 and 1. A threshold (usually 0.5) is then used to assign the input to one of the two classes.

**2. The Logistic Function (Sigmoid)**
* **Why it's used:** The core of Logistic Regression is the logistic function (also called the sigmoid function). **This function takes any real-valued number as input and transforms it into a value between 0 and 1. This is perfect for modeling probabilities.**
* **Equation:** $$\text{sigmoid}(z) = \frac{1}{1 + e^{(-z)}}$$where:
    * $z$ is the linear combination of the input features and their weights (plus a bias term): $z = w_1 . x_1 + w_2. x_2 + ... + w_n.x_n + b$
    * $e$ is the base of the natural logarithm (Euler's number, approximately 2.71828)

* **Shape:** The sigmoid function has an S-shaped curve. It approaches 0 as $z$ goes to negative infinity and approaches 1 as $z$ goes to positive infinity.  It crosses 0.5 at $z = 0$.
![[Screenshot 2025-02-28 at 9.44.51 PM.png|400]]
### Characteristics of the Sigmoid Function
1. **S-Shaped Curve**: The most prominent characteristic of the sigmoid function is its S-shaped curve, which makes it suitable for modeling the probability of binary outcomes.
2. **Bounded Output**: The sigmoid function always produces values between 0 and 1, which is ideal for representing probabilities.
3. **Symmetry**: The sigmoid function is symmetric around its midpoint at _x_=0.5.
4. **Differentiability**: The sigmoid function is differentiable, allowing for the calculation of gradients necessary for optimization algorithms like gradient descent.

**3. Mathematical Formulation**
* **Linear Combination:** First, calculate a linear combination of the input features (independent variables) and their corresponding weights: $z = w_1 . x_1 + w_2. x_2 + ... + w_n.x_n + b$
    where:
    * $x_1, x_2, ..., x_n$ are the input features
    * $w_1, w_2, ..., w_n$ are the weights (coefficients) associated with each feature
    * $b$ is the bias (intercept)

* **Sigmoid Transformation:**  Pass the linear combination $z$ through the sigmoid function to get the predicted probability: $$\hat{y}_i =\text{sigmoid}(z_i) = \sigma(z_i) = \frac{1}{1 + e^{(-z_i)}}$$where $\hat{y}$ is the predicted probability of belonging to the positive class (usually class 1), and  $z_i = w^T.x_i + b$

* **Classification:** Apply a threshold to the probability to classify the input:
    * If $\hat{y}$ >= threshold, predict class 1 (positive class)
    * If $\hat{y}$ < threshold, predict class 0 (negative class)

    The default threshold is often 0.5, but you can adjust it based on the specific problem and desired trade-off between precision and recall.

**4. Cost Function (Loss Function)**
* **Why we need it:** During training, we need to adjust the weights ($w$) and bias ($b$) to minimize the difference between the predicted probabilities and the actual class labels. This is done using a cost function.
* **Log Loss (Binary Cross-Entropy):** The most common cost function for Logistic Regression is the log loss (also known as **binary cross-entropy**). It penalizes the model more heavily for confident but wrong predictions. 
* **Equation:** $$J(w, b) = \frac{-1}{m} \sum_i[y_i . log(\hat{y}_i) + (1 - y_i) . log(1 - \hat{y}_i)]$$where:
    * $J(w, b)$ is the cost function (to be minimized)
    * $m$ is the number of training examples
    * $y_i$ is the actual class label (0 or 1) for the $i^{th}$ example
    * $\hat{y}_i$ is the predicted probability of the $i^{th}$ example belonging to class 1
    * $Σ$ represents the sum over all training examples

**Explanation:**
- If $y_i = 1$ (actual positive), the cost is $-log(\hat{y}_i)$. We want $\hat{y}_i$ to be close to 1, so $-log(\hat{y}_i)$ is small (close to 0). If $\hat{y}_i$ is close to 0, the cost becomes very large.
- If $y_i = 0$ (actual negative), the cost is $-log(1 - \hat{y}_i)$. We want $p_i$ to be close to 0, so $(1 - \hat{y}_i)$ is close to 1, and $-log(1 - \hat{y}_i)$ is small. If $\hat{y}_i$ is close to 1, the cost becomes very large.

**5. Optimization (Gradient Descent)**
* **Goal:** To find the values of `w` and `b` that minimize the cost function $J(w, b)$.
* **Gradient Descent:**  An iterative optimization algorithm that updates the weights and bias in the direction of the steepest descent of the cost function.
* **Update Rules:** $$w_j := w_j - \alpha . (\frac{\partial{J}}{\partial{w_j}})$$   (for each weight $w_j$) $$b := b - α . (\frac{\partial{J}}{\partial{b}})$$
where:
- $w_j$ is the $j^{th}$ weight
- $b$ is the bias
- $\alpha$ is the learning rate (controls the step size)
- $(∂J/∂w_j)$ is the partial derivative of the cost function with respect to the $j^{th}$ weight
- $(∂J/∂b)$ is the partial derivative of the cost function with respect to the bias

* **Calculating the Gradients:** $$\frac{\partial{J}}{\partial{w_j}} = \frac{1}{m} \sum [(\hat{y}_i - y_i)  x_{ij}]$$$$\frac{\partial{J}}{\partial{b}} = \frac{1}{m} \sum (\hat{y}_i - y_i)$$
where  $x_{ij}$ is the value of the $j^{th}$ feature for the $i^{th}$ example.

* **Process:**
1.  Initialize $w$ and $b$ with random values (or 0).
2.  Calculate the predicted probabilities $p_i$ for all training examples.
3.  Calculate the cost function $J(w, b)$.
4.  Calculate the gradients $∂J/∂w_j$ and $∂J/∂b$.
5.  Update the weights and bias using the update rules.
6.  Repeat steps 2-5 until the cost function converges (i.e., the cost stops decreasing significantly).

**6. Assumptions of Logistic Regression**
* **Binary Outcome:** The dependent variable must be binary (two categories). If you have more than two categories, you need to use extensions like Multinomial Logistic Regression (one-vs-rest) or Softmax Regression.
* **Linearity:** Logistic Regression assumes a linear relationship between the independent variables and the *log-odds* of the outcome. Log-odds is $\text{log}(\frac{p}{1-p})$. This is often checked by examining scatter plots of the independent variables against the log-odds of the dependent variable.
* **Independence of Errors:** The errors (residuals) should be independent of each other.
* **Little to No Multi-collinearity:** The independent variables should not be highly correlated with each other (multicollinearity). Multicollinearity can inflate the variance of the estimated coefficients and make it difficult to interpret the results.
* **Sufficiently Large Sample Size:** Logistic Regression generally requires a sufficient amount of data for stable and reliable estimates. A rule of thumb is to have at least 10 events (cases where y=1) per predictor (independent variable).

**7. Implementation Considerations**
* **Feature Scaling:** It's important to scale the input features (e.g., using standardization or min-max scaling) before training Logistic Regression. This helps gradient descent converge faster and prevents features with larger ranges from dominating the optimization process.
* **Regularization:** Regularization techniques (L1, L2) are often used to prevent overfitting, especially when you have a large number of features or limited data.
* **Learning Rate:** Choosing an appropriate learning rate (`α`) is crucial. If the learning rate is too large, gradient descent may overshoot the minimum and diverge. If it's too small, gradient descent may converge very slowly. Techniques like learning rate decay or adaptive learning rate methods (e.g., Adam) can help address this.
* **Threshold Adjustment:** While 0.5 is a common threshold, you can adjust it based on the specific application and the desired trade-off between precision and recall. For example, in medical diagnosis, you might want a lower threshold to increase sensitivity (recall) and avoid missing positive cases, even if it means a higher false positive rate.
* **Data Imbalance:** If the classes are highly imbalanced (e.g., 90% negative, 10% positive), accuracy can be misleading. Techniques like oversampling (e.g., SMOTE), under-sampling, or using different evaluation metrics (e.g., precision, recall, F1-score, AUC) are important.

**8. Extensions of Logistic Regression**
* **Multinomial Logistic Regression (Softmax Regression):** Used for multi-class classification (more than two classes). It predicts the probability of belonging to each class and uses a softmax function to normalize the probabilities to sum to 1.
* **Ordinal Logistic Regression:** Used when the dependent variable is ordinal (has a meaningful order, e.g., low, medium, high).
* **Regularized Logistic Regression:** L1 (Lasso) or L2 (Ridge) regularization can be added to the cost function to prevent overfitting and perform feature selection (L1).

**9. Advantages of Logistic Regression**
* **Easy to Interpret:** The coefficients can be interpreted as the change in the log-odds of the outcome for a one-unit change in the predictor variable (holding other variables constant).
* **Efficient:** Computationally efficient to train and use.
* **Provides Probabilities:** Outputs predicted probabilities, which can be useful for decision-making.
* **Well-Understood:** A well-established and widely used algorithm.

**10. Disadvantages of Logistic Regression**
* **Assumes Linearity:** The assumption of linearity can be a limitation in some cases.
* **Binary Outcome:** Only directly handles binary classification problems. Extensions are needed for multi-class problems.
* **Sensitive to Multicollinearity:** Can be affected by multicollinearity.
* **Can Underperform with Complex Relationships:** May not perform well if the relationships between the features and the outcome are highly non-linear or complex.

**11. Evaluation Metrics**
* **Accuracy:** The proportion of correctly classified instances. Can be misleading with imbalanced datasets.
* **Precision:** The proportion of true positives among all predicted positives.  (True Positives / (True Positives + False Positives))
* **Recall (Sensitivity):** The proportion of true positives that are correctly identified. (True Positives / (True Positives + False Negatives))
* **F1-Score:** The harmonic mean of precision and recall. Provides a balanced measure of performance.  (2 * Precision * Recall) / (Precision + Recall)
* **AUC (Area Under the ROC Curve):** Measures the ability of the classifier to distinguish between classes. A higher AUC indicates better performance.
* **Confusion Matrix:** A table that summarizes the performance of the classifier by showing the counts of true positives, true negatives, false positives, and false negatives.
* **Log Loss (Cross-Entropy Loss):** The cost function used during training can also be used as an evaluation metric to assess the model's performance on unseen data.

**12. Derivation of loss functions and gradient descent update formulae**
