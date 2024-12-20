![[Screenshot 2022-12-27 at 12.36.35 AM.png]]

-   Gradient descent can be used to minimize any function, but here we will use it to minimize our cost function for linear regression.
-   Let’s outline on a high level, how this algorithm works:
    -   Start with some parameters $w, b$.
    -   Computes gradient using a single Training example.
    -   Keep changing the values for w,bw,b to reduce the cost function J(w,b)J(w,b).
    -   Continue until we settle at or near a minimum. Note, some functions may have more than 1 minimum.

Implementation:
![[Screenshot 2022-12-27 at 1.26.37 AM.png]]

-   Above is the gradient descent algorithm. Note the learning rate alpha, it determines how big of a step you take when updating w or b.

Intuition:
-   How do we know we are close to the local minimum? Via a game of hot and cold because as we get near a local minimum, the derivative becomes smaller. Update steps towards the local minimum become smaller, thus, we can reach the minimum without decreasing the learning rate.
-   How can we check gradient descent is working correctly?
    -   We can have 2 ways to achieve this. We can plot the cost function J, which is calculated on the training set, and plot the value of J at each iteration (aka each simultaneous update of parameters w,bw,b) of gradient descent.
    -   We can also use an Automatic convergence test. We choose an ϵ to be a very small number. If the cost JJ decreases by less than ϵϵ on one iteration, then you’re likely on this flattened part of the curve, and you can declare convergence.

![[Screenshot 2022-12-27 at 1.32.22 AM.png]]

#### Learning Rate
-   If the learning rate is too small, you end up taking too many steps to hit the local minimum which is inefficient. Gradient descent will work but it will be too slow.
-   If the learning rate is too large, you may take a step that is too big as miss the minimum. Gradient descent will fail to converge.
-   How should we choose the learning rate then? A few good values to start off with are 0.001,0.01,0.1,10.001,0.01,0.1,1 and so on.
-   For each value, you might just run gradient descent for a handful of iterations and plot the cost function J as a function of the number of iterations.
-   After picking a few values of the learning rate, you may pick the value that seems to decrease the learning rate rapidly.![[Screenshot 2022-12-27 at 1.42.14 AM.png]]

Important thing to think about:
- When your $J(w)$ is already at a local minima, the value $w$ is not updated further. This means that if you're already at a local minimum, gradient descent leaves W unchanged. Because it just updates the new value of W to be the exact same old value of W.
- So if your parameters have already brought you to a local minimum, then further gradient descent steps to absolutely nothing. It doesn't change the parameters which is what you want because it keeps the solution at that local minimum. This also explains why gradient descent can reach a local minimum, even with a fixed learning rate alpha.![[Screenshot 2022-12-27 at 1.44.57 AM.png]]
- as we get nearer a local minimum gradient descent will automatically take smaller steps. And that's because as we approach the local minimum, the derivative automatically gets smaller. And that means the update steps also automatically gets smaller. Even if the learning rate alpha is kept at some fixed value.