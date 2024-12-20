Terms: 
1. Aritificial General Intelligence (AGI)

Machine Learning:
The field of study that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel

Types:
1. Supervised
2. Unsupervised
3. Recommender Systems
4. Reinforcement Learning

<mark style="background: #D2B3FFA6;">Supervised</mark>

X -> Y
input to output mapping
given data, corresponding labels -> predict labels for never-foreseen data (brand new data)
Learns from being given "right answers"

examples:
email -> spam? (0/1) = spam filtering

audio -> text transcripts = speech recognition

English -> Spanish = machine translation

ad, user info -> click? (0/1) = online advertising

image, radar info -> position of other cars = self-driving cars

image of phone -> defect? (0/1) = visual inspection

Types:
Regression: predicts a number out of infinitely many possible outputs

Classification: Predicts from a small finite possible outputs like 0/1 (which class it belongs to). predict categories.

<mark style="background: #D2B3FFA6;">Unsupervised</mark>

Data comes only with inputs x, but not outputs y. Algorithms has to find structure/ pattern in unlabeled data.

Given data which is not associated with any output labels.
Job is to find some structure/ pattern in unlabeled data.

Clustering: cluster similar data points together in groups/ clusters
Eg: google news, DNA microarray, grouping customers

Anamoly detection: For unusual data points
Dimensionality reductions: Compress data using fewer numbers.

***
Topics covered:
[[Linear Regression Model]]
[[Cost Function]]
[[Gradient Descent]]
[[Gradient Descent for Linear Reg]]
[[Multi Variate Linear Regression]]
[[Vectorization]]
[[Gradient Descent for Multiple Linear reg]]
[[Feature Scaling]]
