###### <mark style="background: #D2B3FFA6;">Definition</mark>
Data sets where the ratio of positive to negative examples is very far from 50-50 are called skewed data sets. 

###### <mark style="background: #D2B3FFA6;">Why skewed datasets are bad?</mark>
Example 1: For medical diagnosis, if 99 percent of patients don't have a disease, then an algorithm that predicts no one ever has a disease will have 99 percent accuracy or speech recognition. 

Example 2: If a manufacturing company makes smartphones, hopefully, the vast majority of them are not defective. If 99.7 percent have no defect and are labeled y equals 0 and only a small fraction is labeled y equals 1, then `print("0")`, which is not a very impressive learning algorithm. We achieve 99.7 percent accuracy.

Therefore, accuracy isn't the best performance metric when the dataset is skewed. Confusion Matrix is more useful in these cases.

### <mark style="background: #D2B3FFA6;">Confusion Matrix: Precision and Recall</mark>
1. A confusion matrix is a matrix where one axis is labeled with the actual label, is the ground truth label, y equals 0 or y equals 1 and whose other axis is labeled with the prediction.
2. We fill in with each of these four cells, the total number of examples say the number of examples in your dev set in your development set to fell into each of these four buckets.

For example, 
1. if we have 905 examples in our development set with ground truth $y = 0$. 
2. These are called True Negatives (TN) because they were actually negative and the algorithm predicted them as negative. 
3. True positives (TP) because they were actually positive and the algorithm predicted them as positive, let's say we have 68 of these.
4. False negatives (FN) where the algorithm thought it was negative but it was actually positive, let's say we have 18 of these.
5. False positives (FP) where the algorithm thought it was positive but it was actually negative, let's say we have 9 of these.
![[Screenshot 2023-01-14 at 1.33.11 PM.png|500]]

- Precision answers the question "What proportion of positive identifications was actually correct?"
![[Screenshot 2023-01-14 at 1.34.55 PM.png|200]]
- Recall answers the question "What proportion of actual positives was identified correctly?"
 ![[Screenshot 2023-01-14 at 1.35.14 PM.png|200]]

If we consider the [example 2](<Example 2: If a manufacturing company makes smartphones, hopefully, the vast majority of them are not defective. If 99.7 percent have no defect and are labeled y equals 0 and only a small fraction is labeled y equals 1, then `print("0")`, which is not a very impressive learning algorithm. We achieve 99.7 percent accuracy.>) given above, we get 0 recall and either a very small value or undefined precision. The confusion matrix will look like 

![[Screenshot 2023-01-14 at 3.58.42 PM.png]]
* Small precision is not usually that useful if recall is 0.

##### <mark style="background: #D2B3FFA6;">Combining precision and recall - $F_1$ Score</mark>
- We use Precision and Recall as performance metrics for skewed datasets, but sometimes we have one model with higher precision and another with higher recall. How do we compare then?
- We use $F_1$ score. It is the harmonic mean of precision and recall.
![[Screenshot 2023-01-14 at 4.04.41 PM.png]]
						$F_1 Score = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$


