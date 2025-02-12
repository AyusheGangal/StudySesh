
<mark style="background: #ADCCFFA6;">What are the benefits of Machine Learning?</mark>
To summarize, Machine Learning is great for:
- Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.


<mark style="background: #ADCCFFA6;">Define feature engineering vs feature extraction</mark>
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

  
<mark style="background: #ADCCFFA6;">Define Types of Machine Learning based on:</mark>
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


<mark style="background: #ADCCFFA6;">Define Out-of-core learning.</mark>
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

<mark style="background: #ADCCFFA6;"> Define Sampling noise and Sampling bias?</mark>
It is crucial to use a training set that is representative of the cases you want to generalize to. This is often harder than it sounds: if the sample is too small, you will have sampling noise (i.e., non-representative data as a result of chance), but even very large samples can be non-representative if the sampling method is flawed. This is called sampling bias.

**Sampling Noise** refers to the random variations that occur in a sample due to chance when selecting a subset from a population. It arises because different samples may contain different observations, leading to slight fluctuations in results. This noise decreases as the sample size increases.

**Sampling Bias** occurs when the sample is not representative of the overall population due to a flawed selection process. This leads to systematic errors where certain groups are overrepresented or underrepresented, distorting the conclusions drawn from the data.

In short, sampling noise is due to randomness, while sampling bias is due to a flawed sampling method.

<mark style="background: #ADCCFFA6;">Define Overfitting. How can we prevent it?</mark>
**Overfitting** in machine learning refers to a model's tendency to learn not only the underlying pattern in the training data but also the noise, idiosyncrasies, and random fluctuations present in the dataset. This results in a model that has high variance—performing exceptionally well on the training set but failing to generalize to unseen data. Mathematically, overfitting occurs when a model minimizes empirical risk (training error) to an extreme degree, at the cost of increasing generalization error.

Formally, given a dataset D and a model f(x;θ), the model is said to overfit if the training loss is significantly lower than the validation loss

A large gap between these losses indicates poor generalization.

**Techniques to Mitigate Overfitting**
To improve a model's generalization ability, various strategies can be employed:
1. **Regularization:**
    - **L1 Regularization (Lasso):** Adds λ∑∣θi∣\lambda \sum |\theta_i| to the loss function, inducing sparsity.
    - **L2 Regularization (Ridge):** Adds λ∑θi2\lambda \sum \theta_i^2, preventing excessively large weights.
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

<mark style="background: #ADCCFFA6;">Define Regularization.</mark>
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
    - Uses the **L1 norm**: $R(θ)=∑|θ_i|$
    - Encourages sparsity by **driving some weights to zero, leading to feature selection**.
    
2. **L2 Regularization (Ridge Regression)**
    - Uses the **L2 norm**: $R(θ)=∑θ^2_{i}$
    - **Prevents large weight values, making the model more stable.**
    
3. **Elastic Net**
    - A combination of L1 and L2 regularization: $R(θ)=α∑|θ_i|+(1−α)∑θ^2_i$
    - Useful when both feature selection and weight shrinkage are desired.
    
4. **Dropout (Neural Networks)**
    - Randomly drops a fraction of neurons during training to prevent over-reliance on specific features.
    
5. **Early Stopping**
    - Stops training when validation error starts increasing to prevent overfitting.


<mark style="background: #ADCCFFA6;">Differentiate between parameters and hyperparameters.</mark>

| Feature             | **Parameters**                                                                   | **Hyperparameters**                                                               |
| ------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Definition**      | Internal variables learned by the model from the training data.                  | External configurations set before training to control the learning process.      |
| **Examples**        | Weights (θ) and biases in neural networks, coefficients in linear regression.    | Learning rate, number of hidden layers, regularization strength, batch size.      |
| **Learned or Set?** | Learned automatically during training via optimization (e.g., gradient descent). | Set manually or tuned using techniques like grid search or Bayesian optimization. |
| **Scope**           | Specific to the dataset and changes dynamically during training.                 | Remains fixed during training but can be optimized over multiple runs.            |
| **Effect**          | Directly affects model predictions.                                              | Controls how the model learns, influencing convergence speed and generalization.  |

<mark style="background: #ADCCFFA6;">Define No Free Lunch Theorem.</mark>
The **No Free Lunch (NFL) Theorem** in machine learning states that no single algorithm performs best for all possible problems. It implies that an algorithm that works well on one type of task may perform poorly on another.

**Formal Understanding**
For any two learning algorithms A and B, their average performance across all possible problems is the same. This means that without prior knowledge about the specific problem, no one algorithm is inherently superior.

**Implications in Machine Learning**
- **Algorithm Selection**: There is no universally best model; choosing the right one depends on the problem and data.
- **Hyperparameter Tuning**: The best settings vary across datasets.
- **Bias-Variance Tradeoff**: Simpler models generalize better for some tasks, while complex models excel in others.

This theorem highlights the importance of **domain knowledge, feature engineering, and problem-specific experimentation** in ML.

<mark style="background: #ADCCFFA6;">What can go wrong if you tune hyperparameters using test set?</mark>
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


<mark style="background: #ADCCFFA6;">You are given an imbalanced dataset where the minority class constitutes only **1%** of the total data. You train a highly complex deep learning model, and it achieves **99% accuracy** on the test set.</mark>

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

<mark style="background: #ADCCFFA6;">Define upstream and downstream teams?</mark>
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

