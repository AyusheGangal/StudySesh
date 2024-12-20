Linear algebra has applications in many fields of science and technology, and machine learning is certainly not the exception. In fact, it is not a stretch to say that linear algebra is the most useful and widespread math field in machine learning. 

In this video, you'll learn about some of the best applications of linear algebra in machine learning starting from the most popular application that you know by now, linear regression.

A common machine learning approach to modeling systems is called linear regression.

Linear regression is a supervised machine learning approach, which means you've already collected data on many inputs and an output, and your goal is to discover the relationships between them.

![[Screenshot 2024-08-30 at 10.11.31 PM.png]]

For example, suppose you want to predict the electrical power output from a wind turbine. If you had just one feature, like in this case, wind speed shown here on the x-axis, which is the horizontal axis and you plot your target of power output on the y-axis, which is the vertical axis. Then the data points here are representing real measurements of wind speed and power output.

With a model like this, you're making the assumption that this relationship is literally linear. That it can be modeled by a line.

in reality, you have many records in your dataset. You have many equations you could write down like this one for each record in your data. If you add a superscript 1 with parenthesis up here on everything in this first equation I wrote down, then I can write down the same thing for the second example in the dataset and denote that with a superscript 2 in parenthesis like this, and then so on down to the superscript m in parenthesis for the last example, in a dataset containing M records. 

This is called a system of linear equations.

![[Screenshot 2024-08-30 at 10.08.11 PM.png]]

Another example, 
- Imagine you have a dataset containing a series of features, things like wind speed, temperature, atmospheric pressure, humidity, and so on. 
- I call this $x_1$, $x_2$ and so on, up to $x_n$ for a dataset with n features. Then I added a superscript to the dataset to denote which row of data a set of features belong to. Then you have the model weights multiplying it feature, which we wrote as $w_1$, $w_2$ and up to $w_n$. Then we also added a biased term b. 
- We said that equals to the target y, which in this case is power output from the wind turbine. 
- An important thing to note about this system is that while the xs and ys are unique in each row, all of these $x^1$ are different from these $x^2$ and so on and all the $y^1$ is different from this $y^2$ on down to $y^m$. The w values and b are all the same across all rows. 
- Again, what you're saying here with a linear model is that there exists some set of values $w_1$, $w_2$ and so on, up to $w_n$, as well as some value b that when multiplied by any of these rows of features and added up like this, will be able to provide you with an estimate of your target y for that row. 

![[Screenshot 2024-08-30 at 10.15.11 PM.png]]

In other words, with this model you are saying, give me a set of xs and I can estimate a value for y because I have a model that tells me what all the w's and b's are. Instead of writing this model out in long form like this, I can instead say I have a vector of weights called w, that is made up of $w_1$, $w_2$ and so on and I multiply that by each row of features x in my matrix of features, which I now call capital X. Then I add a biased term and set that all equal to y, which is a vector of my target variable. Just like that, we're back to a nice simple equation that looks just like the equation of a line.

![[Screenshot 2024-08-30 at 10.45.13 PM.png]]
