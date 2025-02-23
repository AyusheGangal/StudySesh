Gradient Descent is an **optimization algorithm** used to **minimize a loss function** by iteratively updating the model’s parameters in the direction of the steepest descent. It is widely used in machine learning, especially in training models like **Linear Regression, Logistic Regression, and Neural Networks**.
 
In **Linear Regression**, we aim to find the optimal values of w (weights) and b (bias) that minimize the **Mean Squared Error (MSE)**:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$The best parameters $w$ and $b$ minimize this function. Instead of solving it directly using the **Normal Equation**, which is computationally expensive for large datasets, we use **Gradient Descent**, which iteratively improves the model parameters.

Gradient Descent updates the model parameters **in the direction of the steepest decrease** in the loss function. This is done using the **gradient (partial derivative)** of the loss function with respect to each parameter.

The **learning rate** refers to how much the parameters are changed at each iteration. If the learning rate is too high, the model fails to converge and jumps from good to bad cost optimizations. If the learning rate is too low, the model will take too long to converge to the minimum error.

**Computing the Gradient for Linear Regression**
For **Simple Linear Regression** ($y=wx+b$), we compute the **gradients**:

**1. Partial Derivative w.r.t $w$ (Weight Update Rule):**
$$\frac{\partial MSE}{\partial w} = -\frac{2}{m} \sum_{i=1}^{m} x_i (y_i - \hat{y}_i)$$

**2. Partial Derivative w.r.t $b$ (Bias Update Rule):**
$$\frac{\partial MSE}{\partial b} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)
$$

These derivatives tell us **how much to adjust $w$ and $b$ to reduce the error**.

**Gradient Descent Update Rule**
We update the parameters $w$ and $b$ using the **learning rate α**:
$$w:=w−α\frac{∂MSE}{∂w}$$

where:
- α = **learning rate**, which controls the step size in each update.

The **learning rate** refers to how much the parameters are changed at each iteration. If the learning rate is too high, the model fails to converge and jumps from good to bad cost optimizations. If the learning rate is too low, the model will take too long to converge to the minimum error.

The choice of α (learning rate) significantly affects the optimization process:
✅ **Too Small α → Convergence is slow**.  
❌ **Too Large α→ Can overshoot or even diverge**.  
✅ **Optimal α → Finds the minimum efficiently.

📌 **Solution:** Use techniques like **learning rate decay** or **adaptive optimizers** (e.g., Adam, RMSprop).
![[Screenshot 2025-02-16 at 10.03.28 PM.png|500]]
### Stopping criteria for gradient descent:
1. **Loss Change is Small** → If $| L_{t} - L_{t-1} | < \epsilon$, stop.
2. **Gradient is Close to Zero** → If $\nabla L \approx 0$, we are near a minimum.
3. **Max Iterations Reached** → A predefined limit is hit.


### Update formula for MSE derivation (gradient descent formula)
![[Screenshot 2025-02-21 at 7.18.24 PM.png]]
![[Screenshot 2025-02-21 at 7.21.11 PM.png]]

#### Q: Manually calculate gradient descent for this example: x = [1, 2, 3]; y = [2, 2.8, 3.6]
![[Screenshot 2025-02-21 at 7.35.40 PM.png]]
![[Screenshot 2025-02-21 at 7.37.02 PM.png]]


### Gradient descent and normal equation comparison (when to use which?)

| **Method**           | **Computational Cost**      | **Suitable for Large Data?**  | **Requires Learning Rate?** |
| -------------------- | --------------------------- | ----------------------------- | --------------------------- |
| **Gradient Descent** | $O(mn)$ per iteration       | ✅ Yes                         | ✅ Yes                       |
| **Normal Equation**  | $O(n^3)$ (Matrix Inversion) | ❌ No (expensive for high nnn) | ❌ No                        |

For **small datasets**, the **Normal Equation** is fine.  
For **large datasets**, **Gradient Descent** is preferred!

### Types of gradient descent
Gradient Descent is an optimization algorithm used to minimize a function by iteratively adjusting parameters. There are **three main types** of Gradient Descent, each with different computational trade-offs.

**Batch Gradient Descent (BGD)**
- Uses **all training examples** to compute the gradient before updating parameters.
- The update step happens **once per epoch** (one full pass through the dataset).

**Update Rule**$$w := w - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla L(y_i, \hat{y}_i)$$
$$b := b - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla L(y_i, \hat{y}_i))$$
where:
- m = number of training samples
- α = learning rate

**Pros ✅**
✔ More **stable convergence** (consistent updates).  
✔ Works well for **small datasets**.

**Cons ❌**
✖ **Slow** for large datasets.  
✖ Requires **more memory**, as the entire dataset must be loaded at once.


**Stochastic Gradient Descent (SGD)**
- Updates **parameters after each individual training example** rather than the full dataset.
- The update happens **m times per epoch** (once per training sample).

**Update Rule**$$w := w - \alpha \nabla L(y_i, \hat{y}_i)$$ $$b := b - \alpha \nabla L(y_i, \hat{y}_i)$$
**Pros ✅**
✔ Much **faster** for large datasets.  
✔ Can escape **local minima** better than Batch Gradient Descent.  
✔ Works well for **streaming data** (online learning).

**Cons ❌**
✖ High **variance in updates**, making optimization noisy.  
✖ **May not converge smoothly**, oscillating around the minimum.

📌 **Solution:** Use **momentum-based optimizers** (e.g., Adam, RMSprop) to stabilize updates.


**Mini-Batch Gradient Descent (MBGD)**
- Uses **a small subset (batch) of training samples** to compute the gradient.
- The update happens **multiple times per epoch** (once per batch).

**Update Rule**
For batch size $B$:$$w := w - \alpha \frac{1}{B} \sum_{i=1}^{B} \nabla L(y_i, \hat{y}_i)$$ $$b := b - \alpha \frac{1}{B} \sum_{i=1}^{B} \nabla L(y_i, \hat{y}_i)$$
**Pros ✅**
✔ Balances **efficiency and stability** (faster than BGD, smoother than SGD).  
✔ Works well with **modern hardware** (vectorized operations on GPUs).  
✔ Converges **faster than BGD** and **more stable than SGD**.

**Cons ❌**
✖ Requires tuning the **batch size** for optimal performance.  
✖ Still involves **some noise**, but less than pure SGD.

**🛠 Best Practice:**
- Common batch sizes: **32, 64, 128, 256** (powers of 2 for GPU efficiency).


**Comparison of Gradient Descent Types**

|**Type**|**Dataset Size**|**Update Frequency**|**Convergence Stability**|**Speed**|**Memory Usage**|
|---|---|---|---|---|---|
|**Batch GD**|Small|Once per epoch|High (Stable)|Slow|High|
|**SGD**|Large|Every sample|Low (Noisy)|Fast|Low|
|**Mini-Batch GD**|Medium to Large|Every batch|Medium (Balanced)|Medium-Fast|Medium|

| **Type**                              | **Description**                                          | **Pros**                                 | **Cons**                   |
| ------------------------------------- | -------------------------------------------------------- | ---------------------------------------- | -------------------------- |
| **Batch Gradient Descent**            | Computes gradient using **entire dataset**               | Stable convergence                       | Slow for large datasets    |
| **Stochastic Gradient Descent (SGD)** | Computes gradient using **one random sample per update** | Faster updates, good for online learning | High variance in updates   |
| **Mini-Batch Gradient Descent**       | Uses a **small batch of samples** per update             | Balance between stability and speed      | Needs tuning of batch size |
✅ **Mini-batch GD** is the most commonly used variant


### Momentum, RMSprop and Adam (Adaptive Moment Estimation)
To improve standard Gradient Descent, **adaptive optimizers** are used:
🔹 **Momentum** – Accelerates SGD in relevant directions.  
🔹 **RMSprop** – Adapts learning rate per parameter.  
🔹 **Adam (Adaptive Moment Estimation)** – Combines Momentum and RMSprop (widely used).

Standard **Gradient Descent** (SGD, Mini-Batch GD) has limitations, such as slow convergence, high variance, and difficulty escaping saddle points. To address these issues, advanced optimizers like **Momentum, RMSprop, and Adam** are used.


**Momentum Gradient Descent**

Instead of using only the **current gradient**, **Momentum** accumulates the **past gradients**, allowing the update direction to build speed like a moving object. Momentum gradient descent is a variant of gradient descent that **_adds a momentum term to the update rule_**. 

The momentum term accumulates the gradient values over time and dampens the oscillations in the cost function, leading to faster convergence. This is particularly useful in cases where the cost function has a lot of noise or curvature, which can cause traditional gradient descent to get stuck in local minima.

**Update Rule:**$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(w_t)$$$$w_t := w_t - \alpha v_t$$​
where:
- $v_t$​ = exponentially weighted moving average of past gradients (velocity).
- $\beta$ = momentum coefficient (usually **0.9**).
- $\alpha$ = learning rate.
- $\nabla L(w_t)$ = current gradient.

**Advantages:**
✔ Helps **accelerate convergence** by smoothing gradient updates.  
✔ Reduces **oscillations** in steep landscapes (good for deep networks).  
✔ Can **escape local minima** faster.

**Disadvantages:**
✖ Needs **tuning of β.  
✖ Can **overshoot** if β is too high.


**RMSprop (Root Mean Square Propagation)**

RMSprop adapts the **learning rate for each parameter** based on the **moving average of past squared gradients**, preventing oscillations and stabilizing updates.

**Update Rule:**$$s_t = \beta s_{t-1} + (1 - \beta) (\nabla L(w_t))^2$$$$w_t := w_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla L(w_t)$$
where:
- $s_t$ = exponentially weighted moving average of **squared** gradients.
- $\epsilon$ = small constant (prevents division by zero, typically $10^{-8}$).
- $\beta$ = decay rate (typically **0.9**).

**Advantages:**
✔ Helps handle **sparse features** (important in deep learning).  
✔ Reduces oscillations in **steep valleys**.  
✔ Works well for **non-stationary problems** (like reinforcement learning).

 **Disadvantages:**
✖ **Does not include momentum**, so it may be slower in some cases.  
✖ Learning rate adaptation can make it sensitive to **initialization**.


**Adam (Adaptive Moment Estimation)**

Adam **combines Momentum and RMSprop**, adapting learning rates **per parameter** while also using momentum to smooth updates.

**Update Rule:**
1. **Compute first moment (Momentum-like update):** $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w_t)$$
2. **Compute second moment (RMSprop-like update):** $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(w_t))^2$$
3. **Bias correction (to prevent underestimation in early iterations):** $$\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1 - \beta_2^t}$$​​
4. **Parameter update:** $$w_t := w_t - \frac{\alpha}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}​$$
where:
- $m_t$​ = first moment estimate (**Momentum** term).
- $v_t$​ = second moment estimate (**RMSprop** term).
- β$_1$=0.9, β$_2$=0.999 (default values).
- ϵ=$10^{−8}$ (to avoid division by zero).

**Advantages:**
✔ **Combines best of Momentum and RMSprop.**  
✔ Works well for **sparse data** and **non-stationary environments**.  
✔ **Default settings work well**, making it easy to use.

**Disadvantages:**
✖ **Computationally expensive** (requires tracking two moments per parameter).  
✖ May **overshoot local minima** if learning rate is too high.


**Comparison of Optimizers**

| Optimizer    | Uses Momentum? | Adapts Learning Rate? | Pros                                               | Cons                                     |
| ------------ | -------------- | --------------------- | -------------------------------------------------- | ---------------------------------------- |
| **SGD**      | ❌ No           | ❌ No                  | Simple, works well for convex problems             | Can be slow, sensitive to learning rate  |
| **Momentum** | ✅ Yes          | ❌ No                  | Accelerates convergence, reduces oscillations      | Can overshoot, requires tuning β         |
| **RMSprop**  | ❌ No           | ✅ Yes                 | Reduces oscillations, works well for deep networks | No momentum, sensitive to initialization |
| **Adam**     | ✅ Yes          | ✅ Yes                 | Best of Momentum + RMSprop, fast convergence       | Computationally expensive, may overshoot |

 **Summary**
- **Momentum**: Accelerates learning, reduces oscillations.
- **RMSprop**: Adapts learning rate per parameter, good for deep learning.
- **Adam**: Combines **Momentum + RMSprop**, widely used in deep learning.