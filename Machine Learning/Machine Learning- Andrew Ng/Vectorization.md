- In linear algebra, counting starts from 1.
- But in python numpy, it starts from 0.
- $\vec{w} = [w_1 w_2 w_3]$
	b is a number
	$\vec{x} = [x_1 x_2 x_3]$
	here, $n$ = 3
- Using numpy
![[Screenshot 2023-01-04 at 11.59.42 AM.png|300]]
- We access $w$ as,
	- $w[0]$ = 1 etc.

Without Vectorization:
- Manually adding all
![[Screenshot 2023-01-05 at 12.59.16 PM.png|400]]
- Using Sigma notation and a for loop in python
![[Screenshot 2023-01-05 at 1.00.40 PM.png|400]]

With Vectorization:
![[Screenshot 2023-01-05 at 1.13.55 PM.png|200]]

Benefits of Vectorization:
1. it makes code shorter, is now just one line of code. 
2. it also results in your code running much faster than either of the two previous implementations that did not use vectorization. 
	- The reason that the vectorized implementation is much faster is behind the scenes. The NumPy dot function is able to use parallel hardware in your computer and this is true whether you're running this on a normal computer, that is on a normal computer CPU or if you are using a GPU, a graphics processor unit, that's often used to accelerate machine learning jobs. 
	- The ability of the NumPy dot function to use parallel hardware makes it much more efficient than the for loop or the sequential calculation that we saw previously. 
	- Now, this version is much more practical when n is large because you are not typing w0 times x0 plus w1 times x1 plus lots of additional terms like you would have had for the previous version. But while this saves a lot on the typing, is still not that computationally efficient because it still doesn't use vectorization.

Comparison between vectorized and non-vectorized implementation
![[Screenshot 2023-01-05 at 4.33.24 PM.png|500]]
- Let's take a deeper look at how a vectorized implementation may work on your computer behind the scenes. Let's look at this for loop. 
- The for loop like this runs without vectorization. If j ranges from 0 to say 15, this piece of code performs operations one after another. On the first timestamp which I'm going to write as t0. It first operates on the values at index 0. At the next time-step, it calculates values corresponding to index 1 and so on until the 15th step, where it computes that. In other words, it calculates these computations one step at a time, one step after another. 
- In contrast, this function in NumPy is implemented in the computer hardware with vectorization. The computer can get all values of the vectors w and x, and in a single-step, it multiplies each pair of w and x with each other all at the same time in parallel. Then after that, the computer takes these 16 numbers and uses specialized hardware to add them altogether very efficiently, rather than needing to carry out distinct additions one after another to add up these 16 numbers. 
- This means that codes with vectorization can perform calculations in much less time than codes without vectorization. This matters more when you're running algorithms on large data sets or trying to train large models, which is often the case with machine learning.

	 - Example: Gradient Descent
		![[Screenshot 2023-01-05 at 4.34.57 PM.png|500]]




