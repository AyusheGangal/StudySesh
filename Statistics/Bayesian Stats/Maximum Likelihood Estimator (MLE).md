MLE stands for Maximum Likelihood Estimation. It is a statistical method used to estimate the parameters of a probability distribution, based on observed data.

In MLE, we assume a specific probability distribution for the data, and then find the values of the distribution's parameters that maximize the likelihood of observing the data. The likelihood function is a function of the parameters, which gives the probability of observing the data, given the assumed probability distribution.

For example, let's say we have a set of data points that we believe are normally distributed. To estimate the mean and variance of the normal distribution that best fit the data, we can use MLE. We assume that the data is normally distributed, and then find the values of the mean and variance that maximize the likelihood of observing the data.

To find the maximum likelihood estimates of the parameters, we differentiate the likelihood function with respect to each parameter, set the resulting derivatives to zero, and solve for the parameters. In some cases, it may not be possible to solve for the maximum likelihood estimates analytically, and numerical optimization methods may be used instead.

MLE is a widely used method for parameter estimation in various fields, including physics, biology, finance, and engineering. It is a powerful tool for making inferences about the population from a sample of data, and it has many practical applications in data analysis and modeling.


<mark style="background: #D2B3FFA6;">Likelihood is called the inverse probability. </mark>
Likelihood is often referred to as an "inverse probability" because it is related to probability in a way that is opposite to how we typically think about probability.

In probability theory, we are usually interested in calculating the probability of observing a given set of data, given a specific probability distribution. In other words, we start with the probability distribution and calculate the probability of observing the data.

However, in likelihood theory, we are interested in calculating the probability distribution of the parameters of a given probability distribution, given the observed data. In other words, we start with the observed data and calculate the probability distribution of the parameters that could have generated that data.

The key difference between probability and likelihood is that probability is a measure of how likely a specific event is to occur, given the probability distribution, while likelihood is a measure of how likely a particular probability distribution is, given the observed data.

So, while probability measures how likely it is to observe a certain event, given a specific distribution, likelihood measures how likely it is that the distribution generated the observed data. This is why likelihood is often referred to as an "inverse probability".

Bernoulli Likelihood
![[Screenshot 2023-04-18 at 10.55.17 PM.png|400]]

![[Screenshot 2023-04-18 at 10.57.05 PM.png]]

![[Screenshot 2023-04-18 at 10.57.34 PM.png]]

for a small sample, MLE can actually be quite biased.
![[Screenshot 2023-04-18 at 11.00.24 PM.png]]

For the blue curve, we have one value of MLE and for the pink curve, there is a range of values for MLE. This has got to do with the steepness of the function, which corresponds to how confident we are in our estimates. 

Var($\hat{\theta}$) $\alpha$ $\frac{1}{curvature of Likelihood}$
Var($\hat{\theta}$) $\alpha$ $\frac{1}{E[\frac{-\delta^2{Local Likelihood}}{\delta{\theta^2}}]}$

MLE (Maximum Likelihood Estimation) is a method used to estimate the parameters of a probability distribution that best fit the observed data. The general mathematical steps involved in MLE are:

1.  Define a probability distribution with unknown parameters, and write the probability density function or probability mass function for that distribution.
    
2.  Assume that the data is independently and identically distributed (iid) according to this probability distribution.
    
3.  Write down the likelihood function, which is the product of the probability density function or probability mass function evaluated at each observed data point. The likelihood function gives the probability of observing the data, given the assumed probability distribution and the values of the unknown parameters.
    
4.  Take the natural logarithm of the likelihood function to obtain the log-likelihood function. This is a common practice because it simplifies the algebraic expressions and makes it easier to find the maximum likelihood estimates of the parameters.
    
5.  Differentiate the log-likelihood function with respect to each unknown parameter and set the resulting derivatives equal to zero to find the values of the parameters that maximize the likelihood function. These values are the maximum likelihood estimates (MLEs) of the parameters.
    
6.  Check that the second derivative of the log-likelihood function is negative at the maximum likelihood estimates, to ensure that the estimates correspond to a maximum rather than a minimum or a saddle point.
    

In summary, MLE involves selecting a probability distribution, assuming iid observations from that distribution, writing down the likelihood function, taking the natural logarithm of the likelihood function, differentiating the resulting log-likelihood function with respect to the unknown parameters, and finding the values of the parameters that maximize the log-likelihood function. These values are the MLEs of the parameters that best fit the observed data.