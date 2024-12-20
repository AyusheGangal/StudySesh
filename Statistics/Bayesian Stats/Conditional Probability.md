using the same example as before.
![[Screenshot 2023-04-18 at 9.00.52 PM.png|200]]

Joint Prob: P(X=0, Y=1) = 0.1
Conditional Prob: P(X=1|Y=1) = 

Conditional probability is a term used in probability theory and statistics to refer to the probability of an event A occurring, given that another event B has occurred. It is denoted by P(A|B), and it is read as "the probability of A given B".

The formula for conditional probability is:

P(A|B) = P(A and B) / P(B)

where P(A and B) is the probability of both A and B occurring, and P(B) is the probability of event B occurring.

Intuitively, the conditional probability P(A|B) is the probability that event A will occur, if we know that event B has already occurred. It takes into account the additional information provided by the occurrence of event B.

For example, suppose we have a bag containing 4 red balls and 6 blue balls. If we draw one ball at random from the bag, the probability of drawing a red ball is 4/10 or 0.4. However, if we know that the ball drawn is blue, the probability of drawing a red ball on a second draw will be different. Specifically, the probability of drawing a red ball on the second draw, given that the first draw was blue, will be:

P(Red on 2nd draw | Blue on 1st draw) = P(Red and Blue) / P(Blue)

where P(Red and Blue) is the probability of drawing a red ball on the second draw and a blue ball on the first draw, and P(Blue) is the probability of drawing a blue ball on the first draw.

If we assume that the balls are replaced after each draw, then the events of drawing a red ball on the second draw and drawing a blue ball on the first draw are independent. In this case, we have:

P(Red on 2nd draw | Blue on 1st draw) = P(Red) = 4/10 or 0.4

On the other hand, if we assume that the balls are not replaced after each draw, then the events of drawing a red ball on the second draw and drawing a blue ball on the first draw are dependent. In this case, we have:

P(Red on 2nd draw | Blue on 1st draw) = P(Red and Blue) / P(Blue) = (4/10 x 6/9) / (6/10) = 4/15 or 0.2667

This means that the probability of drawing a red ball on the second draw, given that the first draw was blue, is lower than the overall probability of drawing a red ball.![[Screenshot 2023-04-18 at 9.20.25 PM.png]]