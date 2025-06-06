
<mark style="background: #ADCCFFA6;">1. What are the benefits of Machine Learning?</mark>
To summarize, Machine Learning is great for:
- Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.


<mark style="background: #ADCCFFA6;">2. Define feature engineering vs feature extraction</mark>
Both feature engineering and feature extraction are techniques used to improve model performance by creating meaningful input features, but they differ in how they achieve this.

**Feature Engineering**
- Feature engineering is the manual process of selecting, transforming, or creating new features from raw data to improve predictive performance.
- It requires domain knowledge to create meaningful features.
- Often involves mathematical transformations, encoding, and aggregations.
- Focuses on improving model interpretability and accuracy.
- Examples:
	- Creating a new feature BMI = weight / (height^2) in a health dataset.
	- Converting Date of Birth into Age.
	- Encoding categorical variables (e.g., one-hot encoding, label encoding).
	- Extracting text features like word count, sentiment scores, or TF-IDF.
- Used In:
	- Traditional machine learning (Random Forest, XGBoost, etc.).
	- Structured/tabular data problems.

**Feature Extraction**
- Feature extraction is the automated process of transforming raw data into a set of lower-dimensional features while retaining important information.
- Uses mathematical transformations to reduce dimensionality.
- Reduces redundancy and improves efficiency.
- Helps when dealing with high-dimensional data (e.g., images, text).
- Examples:
	- Principal Component Analysis (PCA): Reducing high-dimensional data into principal components.
	- Autoencoders: Using neural networks to compress and reconstruct data.
	- Bag of Words (BoW) / Word Embeddings: Transforming text into numerical representations (e.g., Word2Vec, BERT).
	- SIFT/HOG for extracting key visual features from images.
- Used In:
	- Deep learning (CNNs for image processing, NLP for text).
	- High-dimensional data like images, audio, and text.****

Key Differences:

| Aspect                     | Feature Engineering                     | Feature Extraction                                 |
| -------------------------- | --------------------------------------- | -------------------------------------------------- |
| Process                    | Manual                                  | Automated                                          |
| Goal                       | Create meaningful features              | Reduce dimensionality while preserving information |
| Requires Domain Knowledge? | Yes                                     | No (mostly mathematical transformations)           |
| Common Techniques          | Encoding, transformations, aggregations | PCA, Autoencoders, Word Embeddings                 |
| Used In                    | Traditional ML (tabular data)           | Deep learning, NLP, image processing               |

When to Use What?
- Use Feature Engineering when working with structured/tabular data, where domain knowledge can create meaningful features.
- Use Feature Extraction when dealing with high-dimensional data (e.g., images, text) where you need automated transformation methods.


<mark style="background: #ADCCFFA6;">3. Define Types of Machine Learning based on:</mark>
Based on Human Supervision:
- Supervised Learning: 
	- The algorithm learns from a labeled dataset, which includes input features and corresponding desired outputs (targets or labels).
	- The goal is to learn a mapping function that can accurately predict the output for new, unseen inputs.
	- Examples include classification (predicting categories) and regression (predicting continuous values). Think of it like a student learning with a teacher providing correct answers.

- Unsupervised Learning: 
	- The algorithm learns from an unlabeled dataset, where there are only input features and no target outputs.
	- The goal is to discover hidden patterns, structures, or relationships in the data.
	- Examples include clustering (grouping similar data points), dimensionality reduction (reducing the number of features while preserving important information), and association rule learning (finding relationships between variables). Think of it like a student exploring a subject on their own.

- Semi-supervised Learning: 
	- The algorithm learns from a partially labeled dataset, where some data points have labels and others don't. 
	- It leverages the labeled data to understand the underlying structure and then extends that knowledge to the unlabeled data.
	- This approach is useful when labeling data is expensive or time-consuming. Think of it like a student getting some guidance from a teacher but also learning through self-exploration.

- Reinforcement Learning: 
	- The algorithm learns through trial and error by interacting with an environment. 
	- It receives rewards or penalties based on its actions and aims to learn a policy that maximizes the total reward over time. 
	- This approach is often used in robotics, game playing, and control systems. Think of it like a student learning by doing and receiving feedback.
    

Based on Incremental Learning:
- Online Learning: 
	- The algorithm learns incrementally by processing data instances one at a time or in small batches (mini-batches). 
	- It can adapt to changing data patterns in real-time, making it suitable for dynamic environments. It's like a student continuously learning new things every day.

- Batch Learning: 
	- The algorithm learns by processing the entire dataset at once. 
	- It requires all the data to be available upfront and typically involves a longer training time. It's like a student studying all the material at once before an exam.

Based on Learning Approach:
- Instance-based Learning: 
	- The algorithm learns by memorizing the training data. 
	- When presented with a new input, it finds the most similar instances in the training data and makes a prediction based on those instances. 
	- K-Nearest Neighbors (KNN) is a common example. Think of it like a student answering questions by remembering similar examples they've seen.

- Model-based Learning: 
	- The algorithm learns by building a model from the training data. 
	- This model captures the underlying patterns and relationships in the data. When presented with a new input, the model is used to make a prediction. 
	- Linear Regression, Logistic Regression, and Decision Trees are examples. Think of it like a student understanding the underlying concepts of a subject and using that knowledge to solve new problems.


<mark style="background: #ADCCFFA6;">4. Define Out-of-core learning.</mark>
**Out-of-core learning** is a technique in machine learning that allows you to work with datasets that are too large to fit into your computer's main memory (RAM).

Here's how it works:
- **Divide and Conquer**: The large dataset is broken down into smaller, manageable chunks called mini-batches.
- **Process in Batches**: These mini-batches are loaded into memory one at a time. The machine learning algorithm processes the data in the current mini-batch.
- **Discard and Repeat**: Once a mini-batch is processed, it's discarded from memory to make space for the next one. This process continues until all mini-batches have been processed.

Why is this important?
- **Handles Big Data**: Out-of-core learning is essential for working with massive datasets that are common in today's world. Without it, you'd be limited by your computer's memory capacity.
- **Cost-Effective**: You can process large datasets without needing expensive hardware with huge amounts of RAM.

Key Considerations:
- Disk I/O: Reading data from the disk can be slower than accessing it from memory. Efficiently managing disk input/output operations is crucial for out-of-core learning.
- Algorithm Suitability: Not all machine learning algorithms are easily adaptable to out-of-core learning. Algorithms that require access to the entire dataset at once might not be suitable.

Tools and Libraries:
Several libraries and tools support out-of-core learning, including:
- Scikit-learn: This popular Python library provides out-of-core learning capabilities with algorithms like SGDClassifier and SGDRegressor.
- Dask: A flexible parallel computing library that supports out-of-core computations.
- Vowpal Wabbit (VW): A fast out-of-core learning system for various machine learning tasks

<mark style="background: #ADCCFFA6;"> 5. Define Sampling noise and Sampling bias?</mark>
It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., non-representative data as a result of chance), but even very large samples can be non-representative if the sampling method is flawed. This is called sampling bias.

**Sampling Noise** refers to the random variations that occur in a sample due to chance when selecting a subset from a population. It arises because different samples may contain different observations, leading to slight fluctuations in results. This noise decreases as the sample size increases.

**Sampling Bias** occurs when the sample is not representative of the overall population due to a flawed selection process. This leads to systematic errors where certain groups are overrepresented or underrepresented, distorting the conclusions drawn from the data.

In short, sampling noise is due to randomness, while sampling bias is due to a flawed sampling method.

<mark style="background: #ADCCFFA6;">6. Define Overfitting. How can we prevent it?</mark>
**Overfitting** in machine learning refers to a model's tendency to learn not only the underlying pattern in the training data but also the noise, idiosyncrasies, and random fluctuations present in the dataset. This results in a model that has high variance—performing exceptionally well on the training set but failing to generalize to unseen data. Mathematically, overfitting occurs when a model minimizes empirical risk (training error) to an extreme degree, at the cost of increasing generalization error.

Formally, given a dataset D and a model f(x;θ), the model is said to overfit if the training loss is significantly lower than the validation loss

A large gap between these losses indicates poor generalization.

**Techniques to Mitigate Overfitting**
To improve a model's generalization ability, various strategies can be employed:
1. **Regularization:**
    - **L1 Regularization (Lasso):** Adds $\lambda \sum |w_i|$ to the loss function, inducing sparsity.
    - **L2 Regularization (Ridge):** Adds $\lambda \sum w_i^2$, preventing excessively large weights.
    - **Elastic Net:** A combination of L1 and L2 regularization.
    
2. **Cross-Validation:**
    - **k-Fold Cross-Validation** reduces variance in performance estimates.
    - **Leave-One-Out Cross-Validation (LOOCV)** is useful for small datasets.
    
3. **Early Stopping:**
    - Monitors validation loss and halts training when it starts increasing.
    
4. **Dropout (for Deep Learning):**
    - Randomly disables neurons during training to prevent co-adaptation of weights.
    
5. **Data Augmentation:**
    - Artificially expands the dataset (e.g., image transformations, text paraphrasing).
    
6. **Reducing Model Complexity:**
    - Pruning decision trees or reducing the number of hidden layers/nodes.
    
7. **Ensemble Methods:**
    - **Bagging (e.g., Random Forest):** Reduces variance by training multiple models on bootstrapped samples.
    - **Boosting (e.g., XGBoost):** Combines weak learners iteratively, minimizing bias and variance.

<mark style="background: #ADCCFFA6;">7. Define Regularization.</mark>
**Regularization** is a technique in machine learning used to prevent overfitting by adding a penalty to the model's complexity. It helps improve generalization by discouraging the model from learning overly complex patterns that may not generalize well to unseen data.

If you set the regularization hyper parameter to a very large value, you will get an almost flat model (a slope close to zero); the learning algorithm will almost certainly not overfit the training data, but it will be less likely to find a good solution.

**Mathematical Definition**
Regularization modifies the loss function by adding a regularization term to constrain the model parameters. Given a typical loss function L(θ), regularization introduces a penalty term R(θ), resulting in the new objective function:
$$L_{reg}(θ)=L(θ)+λR(θ)$$
where:
- $L_{reg}(θ)$ is the original loss function (e.g., Mean Squared Error or Cross-Entropy Loss).
- R(θ) is the regularization term, which penalizes large or unnecessary weights.
- λ is a hyperparameter that controls the strength of regularization.

**Types of Regularization**
1. **L1 Regularization (Lasso Regression)**
    - Uses the **L1 norm**: $R(θ)=∑|θ_i|$ (where θ is weight)
    - Encourages sparsity by **driving some weights to zero, leading to feature selection**.
    
2. **L2 Regularization (Ridge Regression)**
    - Uses the **L2 norm**: $R(θ)=∑θ^2_{i}$ (where θ is weight)
    - **Prevents large weight values, making the model more stable.**
    
3. **Elastic Net**
    - A combination of L1 and L2 regularization: $R(θ)=α∑|θ_i|+(1−α)∑θ^2_i$
    - Where θ is weight
    - Useful when both feature selection and weight shrinkage are desired.
    
4. **Dropout (Neural Networks)**
    - Randomly drops a fraction of neurons during training to prevent over-reliance on specific features.
    
5. **Early Stopping**
    - Stops training when validation error starts increasing to prevent overfitting.


<mark style="background: #ADCCFFA6;">8. Differentiate between parameters and hyperparameters.</mark>

| Feature             | **Parameters**                                                                   | **Hyperparameters**                                                               |
| ------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Definition**      | Internal variables learned by the model from the training data.                  | External configurations set before training to control the learning process.      |
| **Examples**        | Weights (θ) and biases in neural networks, coefficients in linear regression.    | Learning rate, number of hidden layers, regularization strength, batch size.      |
| **Learned or Set?** | Learned automatically during training via optimization (e.g., gradient descent). | Set manually or tuned using techniques like grid search or Bayesian optimization. |
| **Scope**           | Specific to the dataset and changes dynamically during training.                 | Remains fixed during training but can be optimized over multiple runs.            |
| **Effect**          | Directly affects model predictions.                                              | Controls how the model learns, influencing convergence speed and generalization.  |

<mark style="background: #ADCCFFA6;">9. Define No Free Lunch Theorem.</mark>
The **No Free Lunch (NFL) Theorem** in machine learning states that no single algorithm performs best for all possible problems. It implies that an algorithm that works well on one type of task may perform poorly on another.

**Formal Understanding**
For any two learning algorithms A and B, their average performance across all possible problems is the same. This means that without prior knowledge about the specific problem, no one algorithm is inherently superior.

**Implications in Machine Learning**
- **Algorithm Selection**: There is no universally best model; choosing the right one depends on the problem and data.
- **Hyperparameter Tuning**: The best settings vary across datasets.
- **Bias-Variance Tradeoff**: Simpler models generalize better for some tasks, while complex models excel in others.

This theorem highlights the importance of **domain knowledge, feature engineering, and problem-specific experimentation** in ML.

<mark style="background: #ADCCFFA6;">10. What can go wrong if you tune hyperparameters using test set?</mark>
Tuning hyperparameters using the **test set** leads to **data leakage** and biased performance estimates. Here’s what can go wrong:

 **1. Overfitting to the Test Set**
- The model indirectly learns patterns specific to the test set rather than generalizing well to unseen data.
- Future performance on truly new data may drop significantly.

**2. Invalid Performance Evaluation**
- The test set should simulate real-world data the model hasn’t seen.
- Using it for tuning means the final accuracy is optimistic and not a reliable measure of generalization.

 **3. Lack of Generalization**
- The model may perform well on this particular test set but fail on a different dataset.
- This is especially dangerous in real-world applications like medical diagnosis or fraud detection.

**Correct Approach: Use a Validation Set**
Instead of tuning on the test set:
1. **Split the data into Training, Validation, and Test sets.**
2. **Use the validation set** to tune hyperparameters.
3. **Use cross-validation** (e.g., k-fold) for better stability.
4. **Use the test set only once** for the final unbiased evaluation.


<mark style="background: #ADCCFFA6;">11. You are given an imbalanced dataset where the minority class constitutes only **1%** of the total data. You train a highly complex deep learning model, and it achieves **99% accuracy** on the test set.</mark>

<mark style="background: #ADCCFFA6;">1. Why might this accuracy be misleading?</mark>
<mark style="background: #ADCCFFA6;">2. What alternative evaluation metrics would you use to assess model performance?</mark>
<mark style="background: #ADCCFFA6;">3. If increasing the dataset size is not an option, how would you improve the model’s ability to correctly classify the minority class?</mark>

**1. Why might this accuracy be misleading?**
- The dataset is **highly imbalanced** (99:1 class ratio), so a naive model could predict the majority class (99% of the time) and still achieve **99% accuracy** without actually learning anything meaningful.
- Accuracy is not a good metric for imbalanced datasets because it **doesn’t account for class distribution**.

 **2. What alternative evaluation metrics would you use?**
Instead of accuracy, better metrics for imbalanced data include:
- **Precision & Recall:** Precision ensures correctness of positive predictions, while recall measures the model’s ability to find all positive cases.
- **F1-score:** The harmonic mean of precision and recall, useful when false negatives and false positives are equally important.
- **AUC-ROC (Area Under Curve - Receiver Operating Characteristic):** Measures how well the model distinguishes between classes.
- **AUC-PR (Precision-Recall Curve):** More reliable for imbalanced datasets compared to ROC.

 **3. If increasing dataset size is not an option, how would you improve the model’s ability to correctly classify the minority class?**
Some strategies to handle imbalance:
- **Resampling Techniques:**
    - **Oversampling the minority class** (e.g., SMOTE - Synthetic Minority Over-sampling Technique).
    - **Undersampling the majority class** to balance representation.
- **Cost-sensitive learning:**
    - Adjusting class weights in loss function (e.g., `class_weight='balanced'` in Scikit-learn).
    - Using focal loss to focus on hard-to-classify examples.
- **Anomaly detection approaches:**
    - Treat minority cases as anomalies and train models accordingly.
- **Data Augmentation:**
    - For images, text, or time-series data, generate synthetic variations.
- **Ensemble methods:**
    - Using bagging/boosting (e.g., **Balanced Random Forest, XGBoost with scale_pos_weight**).

<mark style="background: #FF5582A6;">HOT Questions</mark>
<mark style="background: #D2B3FFA6;">1. Bias-Variance Tradeoff in Real-World Models</mark>
You develop two models for a prediction task:
- **Model A** is simple (e.g., a linear regression) and has **high training and test error**.
- **Model B** is a complex deep learning model with **low training error but high test error**.
**Questions:**
<mark style="background: #ADCCFFA6;">1. What issue is Model A facing? What about Model B?</mark>
We have two models:
- **Model A** is simple and has **high training and test error** → Likely suffering from **high bias (underfitting)**.
- **Model B** is complex and has **low training error but high test error** → Likely suffering from **high variance (overfitting)**.
**Model A** faces underfitting because it's too simple to capture the underlying patterns in the data.  **Model B** overfits because it memorizes the training data but fails to generalize.

<mark style="background: #ADCCFFA6;">2. How would you modify Model B to improve its generalization?</mark>
To improve **Model B's generalization**:
- Use **regularization** (L1/L2, dropout for deep learning).
- Increase training data (if possible).
- Use **cross-validation** to tune complexity.
- Apply **early stopping** in deep learning models.

<mark style="background: #ADCCFFA6;">3. Suppose you only have a **small dataset**, which model is preferable? Why?</mark>
If you only have a **small dataset**, **Model A (simpler model)** is preferable because:
- Complex models need **lots of data** to avoid overfitting.
- Simpler models are **more interpretable and stable** with limited data.


<mark style="background: #D2B3FFA6;">2. Curse of Dimensionality in High-Dimensional Data</mark>
A dataset has **10,000 features**, but only **100 samples**. You try training a machine learning model, but it performs poorly.
**Questions:**
<mark style="background: #ADCCFFA6;">1. Why does high dimensionality hurt model performance in this case?</mark>
**High dimensionality causes issues** because:
- **Sparsity**: In high-dimensional space, data points are far apart, making patterns hard to learn.
- **Overfitting**: With more features than samples, models can perfectly memorize training data but fail to generalize.

<mark style="background: #ADCCFFA6;">2. What techniques can you use to mitigate the curse of dimensionality?</mark>
**Techniques to mitigate the curse of dimensionality**:
- **Feature selection** (remove irrelevant or redundant features).
- **Dimensionality reduction**:
    - **PCA (Principal Component Analysis)**: Compresses features while retaining variance.
    - **t-SNE, UMAP**: Useful for visualization.
- **Regularization** (L1/Lasso regression removes unnecessary features).

<mark style="background: #ADCCFFA6;">3. If you suspect that only a few features are important, how would you identify and retain them?</mark>
To **identify important features**:
- Use **feature importance** from models like **Random Forest, XGBoost**.
- Use **statistical tests** (e.g., mutual information, chi-square test).
- Perform **recursive feature elimination (RFE)**.

<mark style="background: #D2B3FFA6;"> 3. Handling Concept Drift in a Changing Environment</mark>
You build a model for predicting customer preferences in an **e-commerce** platform. Over time, user behavior changes, and the model’s performance degrades.
**Questions:**
<mark style="background: #ADCCFFA6;">1. What phenomenon is occurring here? How does it impact ML models?</mark>
Your e-commerce prediction model degrades over time due to **shifting user behavior**. This is **concept drift**, where the relationship between input features and output labels changes over time.
- **Example**: Customer preferences change, making past data outdated.

<mark style="background: #ADCCFFA6;">2. How can you **detect** and **handle** this issue without retraining the entire model from scratch?</mark>
**How to detect and handle concept drift**:
- **Drift detection**:
    - **Monitor model accuracy over time** (sudden drops indicate drift).
    - Use **Kolmogorov-Smirnov (KS) test** or **Jensen-Shannon divergence** to compare feature distributions.
- **Handling drift**:
    - Periodically **retrain the model** on recent data.
    - Use **weighted training** (give recent data higher importance).
    - Use **adaptive models** like online learning algorithms.

<mark style="background: #ADCCFFA6;">3. Would an online learning approach (e.g., incremental learning) be beneficial in this case? Why or why not?</mark>
**Would online learning help?**
- **Yes**, online learning can update models incrementally, avoiding the need for full retraining.
- **Example**: Algorithms like **SGDClassifier, Hoeffding Trees** allow incremental updates.

<mark style="background: #D2B3FFA6;"> 4. Interpretable AI vs. Black Box Models</mark>
A financial institution uses a **deep learning model** to approve or reject loan applications. The model performs exceptionally well but provides **no explanations** for its decisions.

**Questions:**
<mark style="background: #ADCCFFA6;">1. Why is model interpretability important in this scenario?</mark>
A deep learning model is used for **loan approval**, but it’s not explainable. **Why is interpretability important?**
- **Regulations & Ethics**: In financial decisions, users must understand **why** they were rejected.
- **Debugging & Trust**: Explainable models **build trust** and help detect biases.

<mark style="background: #ADCCFFA6;">2. How can you make a deep learning model more interpretable?</mark>
- **Feature importance analysis** (SHAP, LIME).
- **Attention mechanisms** (for NLP tasks).
- **Decision rule extraction** (e.g., RuleFit, Tree-based surrogate models).

<mark style="background: #ADCCFFA6;">3. If you were forced to choose between a slightly less accurate but interpretable model (e.g., logistic regression) and a highly accurate but black-box model (e.g., deep learning), which one would you choose? Why?</mark>
- If **explainability is required (finance, healthcare)** → Choose an **interpretable model** like Logistic Regression or Decision Trees.
- If **accuracy is the priority (e.g., image recognition)** → Deep learning may be preferable.
- A **compromise**: Use an ensemble of interpretable and high-performance models.

<mark style="background: #D2B3FFA6;"> 5. Adversarial Attacks in Deep Learning</mark>
You train a deep learning model for **image classification**. It performs well, but an attacker modifies an image slightly, making the model classify a "panda" as a "gibbon" with high confidence.
**Questions:**
<mark style="background: #ADCCFFA6;">1. Why does the model misclassify the slightly modified image?</mark>
A small perturbation in an image makes the model misclassify it (e.g., "panda" → "gibbon"). 
- Deep learning models rely on **small pixel-level patterns** rather than holistic features.
- **Gradient-based attacks** (like FGSM, PGD) exploit this by adding subtle noise that shifts predictions.

<mark style="background: #ADCCFFA6;">2. What strategies can be used to make the model more robust to adversarial attacks?</mark>
- **Adversarial training**: Train the model on adversarial examples.
- **Defensive distillation**: Smoothen the model’s decision boundary.
- **Input preprocessing**:
    - Use **Gaussian noise, JPEG compression** to remove adversarial perturbations.
    - **Feature squeezing** (reducing precision of pixel values).

<mark style="background: #ADCCFFA6;">3. Do you think adversarial robustness is important in real-world applications? Why or why not?</mark>
- **Yes**, especially in security-critical applications (e.g., self-driving cars, facial recognition, medical AI).
- Ignoring adversarial robustness can lead to serious vulnerabilities in AI systems.

<mark style="background: #ADCCFFA6;">12. Define upstream and downstream teams?</mark>
In a **machine learning** or **software development** workflow, the terms **upstream** and **downstream teams** refer to dependencies in a pipeline or system.

**🔹 Upstream Teams:**
- Teams that produce **data, models, or services** that your team depends on.
- They provide **input** to your work.
- Example:
    - A **data engineering team** (upstream) processes and provides cleaned datasets for the ML team.
    - A **research team** (upstream) develops foundational models used by an applied ML team.

 **🔹 Downstream Teams:**
- Teams that **consume** your outputs.
- They depend on your work to build or enhance their own.
- Example:
    - If you're building an ML model, a **product engineering team** (downstream) might integrate it into an application.
    - A **business analytics team** (downstream) uses your predictions for decision-making.

In short:
- **Upstream** teams provide inputs.
- **Downstream** teams consume your outputs.

<mark style="background: #ADCCFFA6;">13. How would you select a performance measure in Machine Learning?</mark>
**Choosing the Right Performance Measure in ML**
The best performance metric depends on **problem type, data distribution, and business goals**. Here’s how to select one:

 **1️. Regression Problems**
**Goal:** Predict continuous values.  

| Metric                                | When to Use                                         | Formula                                                                                       |
| ------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **MSE (Mean Squared Error)**          | Penalizes large errors, sensitive to outliers       | $$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$                                            |
| **RMSE (Root Mean Squared Error)**    | Similar to MSE, but interpretable in original units | $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$                       |
| **MAE (Mean Absolute Error)**         | Less sensitive to outliers than MSE                 | $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \|y_i - \hat{y}_i\|$$                               |
| **R² (Coefficient of Determination)** | Measures variance explained by the model            | $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$<br> |

💡 **Selection Tip:**
- **MSE/RMSE** if large errors are costly (finance, healthcare).
- **MAE** if outliers shouldn’t have extreme impact.
- **R²** if you want a percentage measure of variance explained.

 **2️. Classification Problems**
**Goal:** Assign categories (e.g., spam vs. not spam).  

| Metric                   | When to Use                                      | Formula                                                                                               | Comments                                                               |
| ------------------------ | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Accuracy**             | Balanced classes, overall correctness            | $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$<br> | Measures overall correctness of the model.                             |
| **Precision**            | False positives costly (e.g., fraud detection)   | $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$<br>                                    | Measures how many predicted positives are actual positives.            |
| **Recall (Sensitivity)** | False negatives costly (e.g., medical diagnoses) | $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$                                           | Measures how many actual positives are correctly identified.           |
| **F1-Score**             | Balance between precision & recall               | $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$      | Balances **Precision** and **Recall**, useful for imbalanced datasets. |
| **AUC-ROC**              | Imbalanced classes, ranking importance           | Area under the ROC curve                                                                              |                                                                        |

💡 **Selection Tip:**
- **Accuracy** if classes are balanced.
- **Precision** if **false positives** are costly (e.g., fraud detection).
- **Recall** if **false negatives** are costly (e.g., cancer detection).
- **F1-Score** if you need a balance between **precision and recall**.
- **AUC-ROC** if you need a ranking-based evaluation for imbalanced data.

**3️. Clustering Problems**
**Goal:** Group similar data points.  

| Metric                        | When to Use                              | Formula                                         |
| ----------------------------- | ---------------------------------------- | ----------------------------------------------- |
| **Silhouette Score**          | Measures how well clusters are separated | $$S(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$ |
| **Davies-Bouldin Index**      | Measures intra-cluster similarity        | Lower is better                                 |
| **Adjusted Rand Index (ARI)** | Compares clustering with ground truth    | Higher is better                                |

💡 **Selection Tip:**
- **Silhouette Score** if you don’t have true labels.
- **ARI** if ground truth labels exist.
- **Davies-Bouldin Index** if you want cluster compactness.

 **Final Selection Guide:**
✅ **Regression?** → MSE, MAE, RMSE, R²  
✅ **Balanced Classification?** → Accuracy  
✅ **Imbalanced Classification?** → Precision, Recall, F1, AUC-ROC  
✅ **Clustering?** → Silhouette Score, ARI

<mark style="background: #ADCCFFA6;">14. Distance measure (norms) for these performance measure.</mark>
In machine learning, **RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error)** are commonly used to measure the distance between two vectors:
1. The **vector of predictions** $\hat{y}$.
2. The **vector of target values** $y$.

**Types of Distance Measures (Norms)**
Different norms can be used to compute distances:

**1️. Euclidean Norm (ℓ₂ Norm) – Used in RMSE**
- **Formula:** 
$$\|A \|_2 = \sqrt{\sum_{i=1}^{n} |v_i|^2}$$
- **Also Called:** ℓ₂ norm.
- **Description:**
    - Measures the **straight-line** (Euclidean) distance between two points.
    - **More sensitive to large errors** (outliers) due to the squaring operation.
    - Used in **Root Mean Squared Error (RMSE)**.

**2️. Manhattan Norm (ℓ₁ Norm) – Used in MAE**
- **Formula:** 
$$\|A \|_1 = \sum_{i=1}^{n} |v_i|$$
- **Also Called:** ℓ₁ norm, **Manhattan norm**.
- **Description:**
    - Measures the distance by summing absolute differences between points.
    - Called the **Manhattan norm** because it represents movement along **orthogonal city blocks** rather than a straight line.
    - Used in **Mean Absolute Error (MAE)**.
    - **Less sensitive to outliers** than RMSE.

 **3️. Generalized ℓₖ Norm (p-Norm)**
- **Formula:**
$$\|A \|_k = \left( \sum_{i=1}^{n} |v_i|^k \right)^{\frac{1}{k}}$$
- **Description:**
    - A generalized norm where **higher values of kk** put more emphasis on **large values**.
    - Special cases:
        - ℓ$_1$ (Manhattan norm) → MAE
        - ℓ$_2$ (Euclidean norm) → RMSE
        - ℓ$_∞$ (Maximum norm) → Takes the **largest** absolute value in the vector

**4️. Special Cases: ℓ₀ and ℓ∞ Norms**

| Norm        | Formula                              | Interpretation                                                                             | Use Case                                                                                                                                                                                   |
| ----------- | ------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ℓ₀ Norm** | ∥A∥ number of non-zero elements in A | **Cardinality norm** (number of nonzero elements). Often used in **sparsity constraints**. |                                                                                                                                                                                            |
| **ℓ∞ Norm** | $$\|A\|_{\infty} = \max_i \|v_i\|$$  | The **ℓ∞ norm** measures the **largest single deviation** from zero in the vector.         | **SVMs** with a **max-margin classifier** often use **ℓ∞ norm** to ensure that the **maximum margin** between classes is as wide as possible, focusing on the most "marginal" data points. |

 **RMSE vs. MAE: Sensitivity to Outliers**
- **RMSE (ℓ₂ norm)**: More sensitive to **outliers**, because squaring errors magnifies large deviations.
- **MAE (ℓ₁ norm)**: More **robust to outliers**, treating all errors equally.

✅ **When to Use RMSE?**
- If **outliers are rare and normally distributed** (e.g., Gaussian noise).
- When **large errors should be penalized more**.
- Common in **regression tasks**.

✅ **When to Use MAE?**
- If the dataset contains **many outliers** or noise.
- When **all errors should contribute equally**.
- Suitable for **robust modeling** (e.g., median-based loss).

<mark style="background: #ADCCFFA6;">15. What is the interpretation of ℓ∞ Norm?</mark>
The **ℓ∞ norm** is often referred to as the **maximum norm**, and it is the maximum **absolute value** of the elements in a vector.

**Mathematical Definition:**
$$\|A\|_{\infty} = \max_i |v_i|$$
where:
- $A=[v_1,v_2,...,v_n]$ is the vector of values.
- $\max_i$​ selects the **largest absolute value** among the vector elements.

**Interpretation:**
- The **ℓ∞ norm** measures the **largest single deviation** from zero in the vector.
- It is often used in **optimization** problems that focus on **worst-case** scenarios, where you want to minimize the largest possible error or deviation.

**Key Points:**
- It focuses on the **most significant** or **largest error** in a vector, completely ignoring the other values.
- **In ML**, it’s useful for **robust optimization** where the largest error is critical to minimize.

**Example Usage:**
- **SVMs** with a **max-margin classifier** often use **ℓ∞ norm** to ensure that the **maximum margin** between classes is as wide as possible, focusing on the most "marginal" data points.

<mark style="background: #ADCCFFA6;">16. What is the interpretation of ℓ₀ Norm?</mark>
The **ℓ₀ norm** measures the number of **nonzero elements** in a vector. It does **not** consider the magnitude of the values, only whether they are zero or not.

**Mathematical Definition:**
∥A∥ = number of non-zero elements in A

where:
- $A = [v_1, v_2, ..., v_n]$ is the vector.
- The norm counts how many elements in $A$ are **not equal to zero**.

**Interpretation & Key Insights**
- **ℓ₀ norm does not follow standard norm properties** because it isn’t continuous or differentiable.
- It is used to measure **sparsity**—how many elements in a dataset, feature vector, or model weights are nonzero.
- The higher the ℓ₀ norm, the **less sparse** the vector (i.e., more nonzero values).
- The lower the ℓ₀ norm, the **sparser** the vector (i.e., more zeros).

**Applications in Machine Learning**
1. **Feature Selection:**
    - ℓ₀ norm is used to find the **minimum number of features** needed to make accurate predictions.
    - In **compressed sensing** and **sparse modeling**, minimizing ℓ₀ helps select the most important features.
2. **Sparsity-Inducing Regularization:**
    - **LASSO (L1 Regularization) approximates ℓ₀** to encourage feature selection in models.
    - Direct ℓ₀ minimization is **NP-hard** (computationally expensive), so it’s often approximated using ℓ₁ norm.
3. **Neural Networks & Pruning:**
    - Used to remove unnecessary neurons or weights to reduce model size.

**Example**
 **Vector:**
A=[3,0,5,0,0,−2], 
∥A∥$_0$=3
(since there are three nonzero elements: 3,5,−2)
$\| A \|_0 = 3 \quad$

✅ **High ℓ₀ norm → Less sparse (more nonzero values).**  
✅ **Low ℓ₀ norm → More sparse (fewer nonzero values).**

<mark style="background: #ADCCFFA6;">17. Talk about Linear Regression in detail.</mark>
Refer to [[Linear Regression]]

<mark style="background: #ADCCFFA6;">18. What is gradient descent? Why do we need it?</mark>
 Refer to [[Gradient Descent notes]]
 
<mark style="background: #D2B3FFA6;">Note: to take care of regularization when calculating derivative of Loss for gradient descent</mark>


<mark style="background: #ADCCFFA6;">19. Differentiate between generative and discriminative classification with examples.</mark>
Machine learning classifiers are broadly categorized into **Generative** and **Discriminative** models. The key difference lies in **how they model data** and **how they make predictions**.

These are two very different frameworks for how to build a machine learning model. Consider a visual metaphor: imagine we’re trying to distinguish dog images from cat images. **A generative model would have the goal of understanding what dogs look like and what cats look like.** You might literally ask such a model to ‘generate’, i.e., draw, a dog. Given a test image, the system then asks whether it’s the cat model or the dog model that better fits (is less surprised by) the image, and chooses that as its label. **A discriminative model, by contrast, is only trying to learn to distinguish the classes (perhaps without learning much about them).** So maybe all the dogs in the training data are wearing collars and the cats aren’t. If that one feature neatly separates the classes, the model is satisfied. If you ask such a model what it knows about cats all it can say is that they don’t wear collars.

**Generative vs. Discriminative Classifiers**
- **Generative Classifiers:** Learn a model of how the data is generated for each class. They estimate the joint probability distribution P(x, y), where x is the data (features) and y is the class label. From P(x, y), you can derive P(y|x) (the probability of a class given the data) using Bayes' theorem. Examples include Naive Bayes, Gaussian Mixture Models (GMMs), and Hidden Markov Models (HMMs).
- **Discriminative Classifiers:** Directly learn the decision boundary between classes or a mapping from input x to class label y. They estimate the conditional probability distribution P(y|x) directly, without explicitly modeling the underlying data distribution. Examples include Logistic Regression, Support Vector Machines (SVMs), and Neural Networks.

<mark style="background: #ADCCFFA6;">20. What are parametric and non-parametric algorithms?</mark>
Machine learning models are often categorized as **parametric** or **non-parametric**, based on how they learn from data and how they handle complexity.

**Parametric Algorithms**
* **Definition:** Parametric algorithms make strong assumptions about the *functional form* or shape of the underlying data distribution. 
* They assume that the data can be adequately represented by a known probability distribution (e.g., normal distribution, binomial distribution). 
* These algorithms then estimate the *parameters* of that assumed distribution from the training data.

*   **Key Characteristics:**
    * **Fixed Number of Parameters:** The number of parameters that need to be learned is fixed and determined *before* training, based on the chosen model and is independent of the number of training examples.
    * **Simplicity and faster:** Often simpler and faster to train, especially with large datasets.
    * **Strong Assumptions:** The performance heavily depends on the correctness of the assumption about the data distribution. If the assumption is wrong, the model can perform poorly.
    * **Generalization:** Once the parameters are learned, the algorithm can generalize well to new data, assuming the underlying distribution remains consistent.
    
    * **Examples:**
        * **Linear Regression:** Assumes a linear relationship between the input features and the output variable.  The parameters are the coefficients and the intercept.
        * **Logistic Regression:** Assumes a logistic function can model the probability of a binary outcome. The parameters are the coefficients.
        * **Naive Bayes:** Assumes features are conditionally independent given the class. The parameters are the means and variances (for Gaussian Naive Bayes) or probabilities (for Multinomial/Bernoulli Naive Bayes).
        * **Perceptron:** Assumes the data is linearly separable. The parameters are the weights.
        
        * **Linear Regression** → Assumes a linear relationship between features.
        - **Logistic Regression** → Uses the sigmoid function for classification.
        - **Naïve Bayes** → Assumes independence between features.
        - **Support Vector Machines (with Linear Kernel)** → Finds a linear decision boundary.
        - **Neural Networks** → Number of parameters is fixed once architecture is chosen.
	
	- **🔹 Advantages:**
	✔ Computationally **efficient** (fast training & inference).  
	✔ Works well when data follows assumed **distribution**.
	
	- **🔹 Disadvantages:**
	✖ **Limited flexibility** → Can underfit complex patterns.  
	✖ Requires **correct assumptions** about data (e.g., normality, independence).

* **Analogy:** Imagine trying to fit a curve through some points. A parametric method would say, "I'm going to assume this is a parabola," and then estimate the coefficients `a`, `b`, and `c` in the equation `y = ax^2 + bx + c`. The number of parameters is always three, no matter how many points you have.

**Non-Parametric Algorithms**
* **Definition:** Non-parametric algorithms make *minimal* assumptions about the underlying data distribution. They do not assume a specific functional form. Instead, they learn the structure of the data directly from the training examples.

* **Key Characteristics:**
    * **Growing Complexity:** The number of parameters that need to be learned grows with the size of the training data. More data often leads to a more complex model.
    * **Flexibility:** Can model more complex and arbitrary data distributions.
    * **Less Assumption-Dependent:** Less susceptible to poor performance due to incorrect assumptions about the data.
    * **Computational Cost:** Can be computationally more expensive to train and use, especially with large datasets.
    
    * **Examples:**
        * **k-Nearest Neighbors (k-NN):** Stores all training data and classifies new data points based on the majority class of their `k` nearest neighbors. The "parameters" are essentially the training data itself.
        * **Decision Trees:** Partition the data into regions based on feature values. The structure of the tree and the split points are learned from the data.
        * **Support Vector Machines (SVMs) with non-linear kernels (e.g., RBF kernel):** Can create complex decision boundaries by mapping data to high-dimensional spaces. The support vectors and kernel parameters are learned from the data.
        * **Neural Networks (in some interpretations):** While neural networks have a fixed architecture, the number of weights and biases can be very large and can adapt to complex data patterns, making them behave somewhat like non-parametric models, especially deep networks.  However, the fixed architecture can also be seen as a parametric aspect.
        
        * **Decision Trees (CART, Random Forest, XGBoost)** → Learn hierarchical splits in data.
        * **K-Nearest Neighbors (KNN)** → Stores all training data and makes predictions based on similarity.
        * **Support Vector Machines (with RBF Kernel)** → Uses a flexible decision boundary.
        * **Gaussian Processes** → Models distributions over functions, allowing for uncertainty quantification.
        
    - **🔹 Advantages:**
	✔ **More flexible** → Can learn complex relationships.  
	✔ **Works well with large datasets** and high-dimensional data.  
	✔ **No strict assumptions** about the data distribution.
	
	- **🔹 Disadvantages:**
	✖ Computationally **expensive** (training & inference can be slow).  
	✖ **Sensitive to noise** → Overfitting is a risk.  
	✖ **Memory-intensive** → Some models (like KNN) store all training data.

* **Analogy:** Imagine trying to fit a curve through some points. A non-parametric method would say, "I'm not going to assume anything about the shape of the curve. I'm going to connect the dots in a flexible way that best fits the data." The more points you have, the more complex the curve can become.

**Summary Table**

| Feature              | Parametric Algorithms                                             | Non-Parametric Algorithms                                      |
| -------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| **Data Assumptions** | Strong assumptions about data distribution (e.g., normality)      | Minimal assumptions about data distribution (learns from data) |
| **Parameters**       | Fixed number of parameters                                        | Number of parameters grows with data size                      |
| **Complexity**       | Simpler, faster training                                          | More complex, potentially slower training                      |
| **Flexibility**      | Less flexible, can underfit if assumptions are wrong              | More flexible, can fit complex data patterns                   |
| **Examples**         | Linear Regression, Logistic Regression, Naive Bayes               | k-NN, Decision Trees, SVMs (with non-linear kernels)           |
| **Training Speed**   | Fast                                                              | Slow (depends on data)                                         |
| **Inference Speed**  | Fast                                                              | Can be slow                                                    |
| **Risk**             | Underfitting                                                      | Overfitting                                                    |
| **Examples**         | Linear Regression, Logistic Regression, Naïve Bayes, SVM (linear) | Decision Trees, KNN, SVM (non-linear), Random Forest           |

**Important Notes:**
* The line between parametric and non-parametric is not always clear-cut. Some algorithms might have aspects of both.
* "Non-parametric" doesn't mean "no parameters." **It means that the number of parameters is not fixed in advance and grows with the size of the data.**
* The choice between parametric and non-parametric methods depends on the specific problem, the amount of data available, and the knowledge (or lack thereof) about the underlying data distribution.
* Overfitting is a risk with non-parametric algorithms, especially with limited data. Regularization techniques are often used to prevent overfitting.

In short, parametric algorithms are like filling in the blanks of a pre-defined form, while non-parametric algorithms build the form themselves based on the data.

<mark style="background: #ADCCFFA6;">20. Write a note on Logistic Regression</mark>
Refer to [[Logistic Regression notes]]

<mark style="background: #ADCCFFA6;">21. What is sigmoid function?</mark>
The **sigmoid function** is a mathematical function that **maps any real number** to a value between **0 and 1**. It is widely used in **Logistic Regression** and **Neural Networks** for probability estimation and activation functions.

**Mathematical Formula**$$\sigma(z) = \frac{1}{1 + e^{-z}}$$​where:
- $z$ is the input (e.g., $z = w^T x + b$ in Logistic Regression).
- $e$ is Euler's number ($\approx 2.718$).
- $\sigma(z)$ is always between **0 and 1**.

**Graph & Interpretation**: The function has an **S-shaped (sigmoid) curve**.
- When $z \to +\infty, \sigma(z) \approx 1$.
- When $z \to -\infty, \sigma(z) \approx 0$.
- When $z=0, \sigma(0) = 0.5$.

 **Why Use Sigmoid?**
✔ **Probability Interpretation** → Outputs a value in (0,1), useful for binary classification.  
✔ **Smooth & Differentiable** → Enables Gradient Descent optimization.  
✔ **Squashes Large Values** → Prevents extreme outputs, stabilizing training.

**Derivative of Sigmoid Function**
To use **Gradient Descent**, we need the derivative:$$\frac{d\sigma(z)}{dz} = \sigma(z) (1 - \sigma(z))$$This is useful for **computing gradients in Logistic Regression and Neural Networks**.

<mark style="background: #ADCCFFA6;">22. How does logistic regression handle outliers? Can it be improved? Explain in detail</mark>

Outliers are **extreme values** in the dataset that significantly **deviate from the rest of the data**. Logistic Regression is **sensitive** to outliers because it **relies on a linear decision boundary**, and extreme values can **distort the optimization process**.

The **decision boundary** in Logistic Regression is determined by the weight parameters ww, which are **optimized using Gradient Descent** to minimize the **Cross-Entropy Loss**:
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
Since Gradient Descent **adjusts weights based on all samples**, **outliers with extreme feature values** can cause:  
✔ **Large gradient updates**, making training unstable.  
✔ **Shifted decision boundaries**, leading to **misclassification of normal points**.  
✔ **Overfitting**, as the model tries to fit to these rare extreme values.

**Example: Outliers Affecting Decision Boundary**
Consider a dataset where most points follow a **clear decision boundary**, but **one outlier is far away**.
**Without Outliers:**
- The decision boundary is well-defined and **splits the data optimally**.

**With Outliers:**
- The model **tries to adjust** for the outlier, shifting the decision boundary **incorrectly**, leading to **wrong classifications**.

**Visualization:**
- **Without outliers →** The boundary is centered.
- **With outliers →** The boundary shifts **toward the outlier**, making normal points misclassified.


**How Can We Improve Logistic Regression to Handle Outliers?**

**1. Apply Robust Regularization (L1/L2)**: Regularization **penalizes large weight values**, preventing extreme changes due to outliers.

**L2 Regularization (Ridge Regression):**
$$J(w, b) = -\sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] + \lambda \sum w^2$$
**Effect:** Prevents the model from giving too much importance to any single feature, reducing the effect of outliers.

**L1 Regularization (Lasso Regression):**
$$J(w, b) = -\sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] + \lambda \sum |w|
$$
**Effect:** Forces some weights to be **exactly zero**, performing feature selection and reducing the influence of outliers.

**2. Use Robust Feature Scaling (Winsorization or Clipping)**  
Outliers cause issues because **feature values vary drastically**. **Standardization or Clipping** can reduce their effect.

[[Winsorizing the data]]
- Replaces extreme values with **percentile-based limits** (e.g., **clipping the top 1% and bottom 1%** of values).
- Prevents outliers from having an **excessive influence**.

**Standardization:**
- Convert all features to **Z-scores**:
$$X' = \frac{X - \mu}{\sigma}$$
- Ensures **outliers don’t dominate feature scaling**.

**3. Detect and Remove Outliers Before Training**  
Instead of modifying the model, **remove extreme values** from training data.

**Use IQR (Interquartile Range):**
- Compute **Q1 (25th percentile) and Q3 (75th percentile)**.
- Define **outlier threshold**:
$$\text{Outlier} \text{ if } X > Q3 + 1.5 \times IQR \text{ or } X < Q1 - 1.5 \times IQR$$


**Use Z-score Filtering:**
- Compute **Z-score** for each feature:
$$Z = \frac{X - \mu}{\sigma}$$
- Remove data points where Z| > 3.

**4. Use a More Robust Classifier (e.g., Tree-Based Models)**  
If outliers **significantly impact Logistic Regression**, consider using:  
✔ **Decision Trees** (handle outliers better).  
✔ **Random Forests** (reduce sensitivity to single points).  
✔ **Support Vector Machines (SVMs)** with robust kernels.


**Summary Table**

|**Method**|**How It Helps?**|
|---|---|
|**L1/L2 Regularization**|Reduces impact of extreme feature values.|
|**Feature Scaling (Winsorization, Standardization)**|Prevents outliers from dominating training.|
|**Outlier Removal (IQR, Z-score)**|Removes extreme data points before training.|
|**Switching to Robust Models (Trees, SVMs)**|Avoids sensitivity to individual outliers.|

**Conclusion**
- Logistic Regression **is sensitive to outliers**, as they distort the decision boundary.
- **Regularization, feature scaling, and outlier detection** can significantly **reduce their impact**.
- **Tree-based models and SVMs** are alternatives if Logistic Regression **struggles too much**.


<mark style="background: #ADCCFFA6;">23. What happens if we remove the sigmoid function and just use a linear model with Cross-Entropy Loss?</mark>
Without sigmoid, the predictions can be **any real number**, and plugging them into Cross-Entropy loss:
$$J(w, b) = -\sum y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})$$
can result in **negative probabilities** or invalid values for log function, leading to training failures.


<mark style="background: #ADCCFFA6;">24. What would happen if we used MSE as the loss function instead of Cross-Entropy?</mark>
Using MSE:$$J(w, b) = \frac{1}{m} \sum (\hat{y} - y)^2$$leads to **slow learning** and **bad gradient behavior** because:
1. **Gradients shrink** → The derivative of sigmoid is small, making updates very slow.
2. **Non-convexity** → The loss function is no longer convex, making gradient descent inefficient.
Cross-Entropy is **better because it is convex and provides well-scaled gradients**.


<mark style="background: #ADCCFFA6;">25. Can Logistic Regression be used for multi-class classification?</mark>
Yes, but standard Logistic Regression is for **binary classification**. For multiple classes, we use:

1. **One-vs-All (OvA)** → Train multiple binary classifiers, one for each class.
2. **Softmax Regression** → Uses the **Softmax function** instead of Sigmoid to assign probabilities to multiple classes.
$$P(y=k | X) = \frac{e^{w_k^T X}}{\sum_{j=1}^{K} e^{w_j^T X}}​$$
**Softmax is the preferred approach** for multi-class problems.


<mark style="background: #ADCCFFA6;">26. How does feature scaling impact Logistic Regression?</mark>
Logistic Regression is **sensitive to feature magnitudes**. If features have vastly different scales, **gradient descent will converge slowly**.  

**Fix:** Standardize features using **Z-score normalization**:
$$X' = \frac{X - \mu}{\sigma}$$

<mark style="background: #ADCCFFA6;">27. What are convex and non-convex loss functions?</mark>
Loss functions are used to measure how well a machine learning model is performing. They can be **convex** or **non-convex**, which affects how easily we can optimize them using gradient descent.

A loss function J(w)is **convex** if it has a **single global minimum**, meaning **gradient descent will always find the best solution**.

A function is convex if its second derivative is **always non-negative**:
$$\frac{d^2 J}{d w^2} \geq 0$$
**Why It’s Important:**  
✔ **Easier optimization** → No local minima, gradient descent always converges to the global minimum.  
✔ **Guaranteed convergence** if the learning rate is properly chosen.

**Examples of Convex Loss Functions**
1. **Mean Squared Error (MSE) - Used in Linear Regression**$$J(w) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$
	- Parabolic shape → Always convex.
	
2. **Mean Absolute Error (MAE)**
$$J(w) = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$
    - Still convex, but not smooth at $y_i = \hat{y}_i$.
    
3. **Binary Cross-Entropy (Log Loss) - Used in Logistic Regression**
    $$J(w) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
    - **Convex** for Logistic Regression → Ensures smooth optimization.


A loss function is **non-convex** if it has **multiple local minima** and possibly **saddle points**, making gradient descent optimization harder.

Non-convex functions **can have multiple local minima**, meaning gradient descent might **get stuck**.

**Examples of Non-Convex Loss Functions**
1. **Loss Functions in Deep Learning (Neural Networks)**
    - Due to **multiple layers and complex activation functions**, neural network loss landscapes are highly **non-convex**.
    - **Example:** Cross-Entropy Loss in Deep Networks.
    
2. **Hinge Loss - Used in SVMs**
    $$J(w) = \sum_{i=1}^{m} \max(0, 1 - y_i w^T x_i)$$
    - **Non-convex when combined with kernel tricks**.
    
3. **Reinforcement Learning Loss Functions**
    - **Policy gradient methods** use loss functions that are non-convex, making convergence difficult.


**Why Does Convexity Matter?**

| **Aspect**           | **Convex Loss**                               | **Non-Convex Loss**             |
| -------------------- | --------------------------------------------- | ------------------------------- |
| **Optimization**     | Easy, always converges                        | Hard, may get stuck             |
| **Gradient Descent** | Always works                                  | Can get stuck in local minima   |
| **Examples**         | MSE, MAE, Cross-Entropy (Logistic Regression) | Deep Learning, SVM with kernels |

✅ **Convex loss functions ensure easy optimization**.  
✅ **Non-convex loss functions require tricks like momentum, adaptive learning rates, or random restarts**.

**Conclusion**
- **Convex Loss (Logistic Regression, Linear Regression)** → **Guaranteed convergence** with gradient descent.
- **Non-Convex Loss (Neural Networks, SVMs, RL)** → Requires **advanced optimization techniques**.


<mark style="background: #ADCCFFA6;">28. Why are non-convex functions used?</mark>
While **convex functions** are easier to optimize, **non-convex functions** are essential in machine learning because they **better model complex relationships** in data. Many powerful models—such as **deep learning, kernel methods, and reinforcement learning**—inherently require non-convex loss functions.

**Why Do We Use Non-Convex Loss Functions?**
1. Real-World Data is Non-Linear
- Many datasets do not have simple **linear relationships**.
- **Convex models (e.g., Logistic Regression, Linear Regression)** fail when data has **complex patterns**.
- **Non-convex models** (e.g., Deep Learning) can **capture intricate relationships**.

2. Deep Learning Uses Non-Convex Loss Functions
- **Neural networks** have multiple **hidden layers**, introducing **non-linear transformations**.
- This makes the **loss function highly non-convex**, with multiple **local minima and saddle points**.
- However, **gradient-based optimization** still works well due to:
    - **Batch Normalization** (helps smooth the loss landscape).
    - **Momentum and Adam optimizer** (helps escape poor local minima).

1. Kernel Methods in SVMs Are Non-Convex
- **Support Vector Machines (SVMs) with non-linear kernels (RBF, polynomial, etc.)** create **non-convex loss surfaces**.
- These kernels allow **better separation of complex data**, making **SVMs competitive with deep learning** in certain tasks.

1. Reinforcement Learning (RL) Uses Non-Convex Loss
- **Policy gradient methods (e.g., REINFORCE, PPO)** involve **stochastic reward functions**, leading to **non-convex optimization**.
- Despite this, **gradient-based methods still work well in RL**, though **convergence is harder**.

1. Feature Selection in Machine Learning Uses Non-Convex Loss
- **L0 regularization** (selecting a fixed number of features) is **non-convex**, but it is useful when we want **sparse models**.
- **L1 regularization (Lasso)** is a convex approximation of this idea.


**How Do We Optimize Non-Convex Loss Functions?**
Since non-convex functions can **trap models in local minima**, we use special optimization techniques:

1. **Momentum-Based Optimization (SGD + Momentum, Adam, RMSprop)**
- Uses **past gradients** to help escape local minima.
- **Adam optimizer** dynamically adjusts learning rates for different parameters.

2. **Random Restarts**
- Train the model multiple times with **different initializations** to avoid poor local minima.

3. **Dropout & Regularization** (for deep learning)
- **Dropout** helps by forcing models to generalize better.
- **L2 Regularization (Weight Decay)** prevents extreme weight updates.

4. **Batch Normalization**
- Smooths the loss surface, making optimization **more stable**.


**Convex vs. Non-Convex: Trade-offs**

|**Aspect**|**Convex Functions**|**Non-Convex Functions**|
|---|---|---|
|**Optimization**|Easier (global minimum)|Harder (local minima)|
|**Computational Complexity**|Faster|Slower|
|**Expressive Power**|Limited (linear patterns)|Higher (complex patterns)|
|**Example Models**|Linear Regression, Logistic Regression|Deep Learning, SVMs with Kernels, RL|
**Conclusion**
- **Non-convex functions allow us to model complex real-world data** that convex functions fail to capture.  
- **Deep Learning, SVMs, and Reinforcement Learning** would not be possible without non-convex optimization.  
- **Despite challenges, advanced optimizers like Adam & Momentum make training effective.**


<mark style="background: #ADCCFFA6;">29. Why is the sigmoid function used instead of other activation functions?</mark>
- Maps values to **[0,1]**, which is useful for probability.  
- Has a **nice derivative**:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$
which simplifies gradient computation.


<mark style="background: #ADCCFFA6;">30. How does a machine learning model like logistic regression behave with imbalanced datasets?</mark>
It tends to predict the **majority class** more often, leading to poor performance.

**Fixes:**
- **Class weighting** → Assign higher weight to the minority class.
- **Oversampling/Undersampling** → Balance the dataset.
- **Use Precision-Recall curves instead of Accuracy**.


<mark style="background: #ADCCFFA6;">31. Compare Logistic Regression with k-NN. When does Logistic Regression perform better?</mark>
**Logistic Regression is better when:**
- The data is **linearly separable**.
- The dataset is **large**, because k-NN is slow.

**k-NN is better when:**
- The decision boundary is **non-linear**.
- There is **no clear mathematical model**.


<mark style="background: #ADCCFFA6;">32. Modify the sigmoid function with a different non-linearity (e.g., tanh, ReLU). How does it affect model performance?</mark>
Replacing the sigmoid function with a different non-linearity like **tanh** or **ReLU** can significantly impact model performance in various ways:

**Tanh Activation**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
- **Effect:** Tanh has a wider range (-1 to 1) compared to sigmoid (0 to 1), leading to zero-centered outputs. This helps in better gradient flow and can lead to faster convergence.
- **Use case:** Often used in hidden layers for better weight updates.

**ReLU Activation**
$$\text{ReLU}(x) = \max(0, x)$$
- **Effect:** Introduces sparsity (since negative values become zero) and helps avoid the vanishing gradient problem. However, it can suffer from the **dying ReLU problem**, where neurons get stuck at zero.
- **Use case:** Preferred in deep networks for efficiency.

**Impact on Model Performance:**
- **Convergence Speed:** ReLU often leads to faster training due to efficient gradient propagation.
- **Gradient Vanishing:** Tanh still has a vanishing gradient issue for large/small inputs but is better than sigmoid.
- **Final Accuracy:** Depends on the dataset; ReLU generally performs better in deep networks, while tanh might be useful in shallower ones.

<mark style="background: #ADCCFFA6;">33. What is KNN?</mark>
The **K-Nearest Neighbors (KNN)** algorithm is one of the simplest and most intuitive **machine learning algorithms** used for both **classification** and **regression** tasks.

KNN is a **lazy learning, instance-based** algorithm that **does not explicitly learn a model**. Instead, it makes predictions by **finding the K closest training samples** to a given test point and using them to determine the output.

**Intuition**
1. Choose a value for **K** (number of neighbors).
2. Find the **K nearest points** from the training data based on a chosen distance metric (e.g., Euclidean distance).
3. **For classification:** Take a **majority vote** among the K neighbors.
4. **For regression:** Compute the **average** (or weighted average) of the K neighbors.

**Key Idea:** "A data point should be classified the same way as its nearest neighbors."


<mark style="background: #ADCCFFA6;">34. Explain the steps for KNN</mark>
**Step 1: Choose K (number of neighbors)**
- If **K is too small** → Model becomes **sensitive to noise** (overfitting).
- If **K is too large** → Model becomes **too smooth** (underfitting).
- **Rule of thumb:** Choose $K \approx \sqrt{N}$​, where N is the number of training samples.

**Step 2: Calculate Distance**
Common distance metrics:
1. **Euclidean Distance (default)**$$d(X_1, X_2) = \sqrt{\sum (X_{1i} - X_{2i})^2}​$$
    - Used for **continuous numeric data**.
    - Measures **straight-line distance** between points.
    
2. **Manhattan Distance**$$d(X_1, X_2) = \sum |X_{1i} - X_{2i}|$$
    - Good for **grid-like structures** (e.g., chessboard moves).
    
3. **Minkowski Distance (Generalized Distance Metric)**$$d(X_1, X_2) = \left( \sum |X_{1i} - X_{2i}|^p \right)^{1/p}$$
    - If p=1p = 1p=1, it is **Manhattan Distance**.
    - If p=2p = 2p=2, it is **Euclidean Distance**.
    
4. **Cosine Similarity (for Text Data)**$$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$
- Measures the **angle** between two vectors, instead of absolute distance.

**Step 3: Find the K Nearest Neighbors**
- Sort training points by **distance to the test point**.
- Select the **K closest points**.

**Step 4: Make Predictions**
- **For classification:** Assign the **most common label** among K neighbors.
- **For regression:** Take the **average** (or weighted average) of neighbors' values

<mark style="background: #ADCCFFA6;">Define tokens and tokenization?</mark>
**Tokens** are small units of data used to train gen AI models like ChatGPT and help them understand and generate language. This data may take the form of whole words, subwords, and other content. A **token** is a **smallest meaningful unit** of text. It can be:  
- A **word** ("apple", "machine")  
- A **subword** ("un-", "-ing")  
- A **character** ("a", "b", "c")  
- A **symbol** ("$", "@", "5")

Tokens are essential for language models because they are the smallest units of meaning. By analyzing tokens, models can understand the structure and semantics of text. The process of making raw data like text trainable for language models is known as **tokenization**. This may include splitting text into individual words.

**Tokenization** is the process of **splitting text into tokens** for processing in NLP (Natural Language Processing).

**Example:**  
**Input:** `"Machine learning is fun!"`  
**Word Tokenization:** `["Machine", "learning", "is", "fun", "!"]`  
**Character Tokenization:** `["M", "a", "c", "h", "i", "n", "e", ...]`

**Types of Tokenization**
- **Word Tokenization:** Splits text into words (`["Hello", "world"]`).  
- **Subword Tokenization:** Breaks words into meaningful parts (`["mach", "ine", "learn", "ing"]`).  
- **Character Tokenization:** Breaks text into individual characters (`["H", "e", "l", "l", "o"]`).  
- **Byte Pair Encoding (BPE):** Merges common character pairs (`["ma", "chine", " learn", "ing"]`).  
- **Sentence Tokenization:** Splits paragraphs into sentences.

**Why is Tokenization Important?**
- Helps **convert text into numerical form** for ML models.  
- Improves **efficiency** by reducing vocabulary size.  
- Affects **accuracy**—poor tokenization can harm model performance.

