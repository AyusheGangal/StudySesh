![[Screenshot 2022-12-27 at 2.04.13 AM.png]]

Derivation for the above formula
![[Screenshot 2022-12-27 at 2.05.05 AM.png]]

Algorithm:
![[Screenshot 2022-12-27 at 2.05.53 AM.png]]

Issue with GD:
1. If we have more than one local minima, GD lead to a local minimum than a global minimum.
2. So depending on different initializations of $w,b$ we can end up with different local minimas.

Solution to this problem:
1. Using 'Squared Error Loss' will never have multiple local minimas, only a single global minimum.
2. This is because it a convex function (a bowl shaped function).
3. When you implement gradient descent on a convex function, one nice property is that so long as you're learning rate is chosen appropriately, it will always converge to the global minimum.

[Medium Article for Different Optimizers:](https://sweta-nit.medium.com/batch-mini-batch-and-stochastic-gradient-descent-e9bc4cacd461)

### Batch Gradient Descent:
- The term Batch grading descent refers to the fact that on every step of gradient descent, we're looking at all of the training examples, instead of just a subset of the training data.

> BGD is a variation of the gradient descent algorithm that calculates the error for each eg in the training datasets, but only updates the model after all training examples have been evaluated.

Let us understand like this,  
suppose I have 'n' number of records. BGD will try to send all data and calculates the summation of loss then do dE/dw (E=summation of loss) i.e., it is going to calculate summation of all the loss and it is going to perform backward propagation based on the summation of the loss.![[Screenshot 2022-12-27 at 2.15.34 AM.png|300]]
Here, batch size = n

One cycle through entire training datasets is called a training epoch. Therefore, it is often said that BGD performs model updates at the end of each training epoch.

**_Advantages_**:
-   It is more computationally efficient.
-   It is a learnable parameter : whenever we are trying to calculate a new weight, we are trying to consider all the data which is available to us based on the summation of the loss. So, we are trying to find out or derive the new value of the weight / bias , which is a learnable parameter.

**_Disadvantages_**:
-   **Memory consumption is too high**: we are trying to send all the data inside the network one by one, so, we need some kind of memory to store a loss which we have received in each and every iterations. Once we are done with passing datasets through the network, we calculate the loss. So this this case , memory consumption will be too high, and this happens in each and every step.
-   If memory consumption is too high, we can say that thr computation will be high and calculation will be very slow and so the optimization will be slower as compared to any other optimizer.

**_Advice_**: try to decrease your batch size.

**_Points_**:
-   BGD tries to converge itself, so it will be able to get a global minima. i.e.,

If , converges **→**global minima(dE/dw=0) and
If, non converges**→**local minima

> **Convergence:** Reaching a point in which gradient descent makes very small changes in your objective function is called convergence, which doesn’t mean it reached the optimal result (but it is really quite quite near, if not on it)

### Adam Optimizer:
![[Screenshot 2022-12-27 at 2.21.00 AM.png]]