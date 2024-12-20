Consider a scenario, where $x_1$ : size (feet$^2$) & $x_2$ : Number of bedrooms
we use this to predict price of house as
$\widehat{price} = w_1*x_1 + w_2*x_2 + b$
	We have, $x_1$ to be large and $x_2$ to be really small.

If we have a training example, where $x_1$ = 2000, $x_2$ = 5 and price = 500K
How should we select the parameters $w_1, w_2$ and $b$?
-  For $w_1$ = 50, $w_2$ = 0.10, $b$ = 50
	$\widehat{price} = 50*2000 + 0.1*5 + 50$
	$\widehat{price} = 100050500$
-  For $w_1$ = 0.1, $w_2$ = 50, $b$ = 50
	$\widehat{price} = 0.1*2000 + 50*5 + 50$
	$\widehat{price} = 500k$

when a possible range of values of a feature is large, like the size and square feet which goes all the way up to 2000. It's more likely that a good model will learn to choose a relatively small parameter value, like 0.1. Likewise, when the possible values of the feature are small, like the number of bedrooms, then a reasonable value for its parameters will be relatively large like 50.
*** 

![[Screenshot 2023-01-05 at 5.06.21 PM.png|600]]

- For this data, If you plot the training data, you notice that the horizontal axis is on a much larger scale or much larger range of values compared to the vertical axis.
- look at how the cost function might look in a contour plot. You might see a contour plot where the horizontal axis has a much narrower range, say between zero and one, whereas the vertical axis takes on much larger values, say between 10 and 100. 
- So the contours form ovals or ellipses and they're short on one side and longer on the other. And this is because a very small change to w1 can have a very large impact on the estimated price and that's a very large impact on the cost J. Because w1 tends to be multiplied by a very large number, the size and square feet. In contrast, it takes a much larger change in w2 in order to change the predictions much. And thus small changes to w2, don't change the cost function nearly as much. 

If we were to run gradient descent:
- and were to use your training data as is. Because the contours are so tall and skinny gradient descent may end up bouncing back and forth for a long time before it can finally find its way to the global minimum.
- ![[Screenshot 2023-01-05 at 8.56.23 PM.png|300]]
- In situations like this, a useful thing to do is to scale the features. This means performing some transformation of your training data so that x1 say might now range from 0 to 1 and x2 might also range from 0 to 1. So the data points now look more like this and you might notice that the scale of the plot on the bottom is now quite different than the one on top.
- 