For Linear regression with one variable

Model: $f_{w,b} (X) = wX + b$
where $w, b$ => parameters/ coefficients/ weights
$b$ = > y-intercept/ offset

Parameters: A model parameter is a configuration variable that is internal to the model and whose value can be estimated from data.

-   They are required by the model when making predictions.
-   They values define the skill of the model on your problem.
-   They are estimated or learned from data.
-   They are often not set manually by the practitioner.
-   They are often saved as part of the learned model.

$\hat{y}^{(i)} = f_{w,b} (x^{(i)})$
$f_{w,b} (x^{(i)}) = wx^{(i)} + b$

Goal: Find w, b such that $\hat{y}^{(i)}$ is close to $y^{(i)}$ for $(x^{(i)}, y^{(i)})$

Cost function (Squared error cost function):
![[Screenshot 2022-12-26 at 8.55.56 PM.png|500]]
- The cost function takes the prediction $\hat{y}^{(i)}$  and compares it to the target $y^{(i)}$ by taking $\hat{y}^{(i)} - y^{(i)}$. 
- This difference is called the error,  we're measuring how far off to prediction is from the target. 
- This error is squared to account for even minute errors. 
- To build a cost function that doesn't automatically get bigger as the training set size gets larger by convention, we will compute the average squared error instead of the total squared error and we do that by dividing by m.
- The extra division by $m$ just makes the later calculations look neater, and the cost function still works.
- Can also be written as:
![[Screenshot 2022-12-26 at 8.57.38 PM.png|400]]

Intuition:
- Our goal is to choose w to minimize the Cost Function wrt w and b
- ![[Screenshot 2022-12-26 at 9.03.48 PM.png|100]]

![[Screenshot 2022-12-26 at 9.05.18 PM.png]]



