https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

### What is the AUC - ROC Curve?
AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes. <mark style="background: #ADCCFFA6;">Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.</mark>

![[Screenshot 2023-01-26 at 12.34.17 AM.png|300]]
The ROC curve is plotted with TPR against the FPR where TPR is on the y-axis and FPR is on the x-axis.

### Types of Error (1 and 2)
-   Type I error occurs when the Null Hypothesis (H0) is mistakenly rejected. This is also referred to as the False Positive Error. 
-   Type II error occurs when a Null Hypothesis that is actually false is accepted. This is also referred to as the False Negative Error.

