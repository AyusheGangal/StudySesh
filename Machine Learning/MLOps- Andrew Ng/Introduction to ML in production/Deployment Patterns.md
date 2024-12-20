### <mark style="background: #D2B3FFA6;">Common deployment cases</mark>
1. New product/ capability
2. Automate/assist with manual task: Shadow mode deployment takes advantage of this
3. Replace previous ML system

#### <mark style="background: #D2B3FFA6;">Key ideas (Recurring themes)</mark>
- Gradual ramp up with monitoring
- Rollback

Example: Visual inspection
- <mark style="background: #ABF7F7A6;">Shadow mode deployment</mark>
	- we start by having an ML system shadows the human inspector and run in parallel with the human inspector. 
	- During this initial phase, the learning algorithm's output is not used for any decision in the factory.
	- The purpose of this type of deployment is that it allows you to gather data on how the learning algorithm is performing in comparison with human judgment.
	- Lets you verify the performance of a learning algorithm.
	
- <mark style="background: #ABF7F7A6;">Canary deployment</mark>
	- Roll out to small fraction (say 5%) of traffic initially, and let the algorithm make real decisions.
	- Gives an opportunity to monitor the system and ramp up traffic gradually.
	- Allows one to spot problems in the early stages
	
- <mark style="background: #ABF7F7A6;">Blue Green deployment</mark>
	- Assuming a system, a camera software for collecting phone pictures in your factory. These phone images are sent to a piece of software that takes these images and routes them into some visual inspection system.![[Screenshot 2023-01-09 at 3.06.08 PM.png]]
	
	- In the terminology of a blue green deployments, the old version of your software is called the blue version and the new version, the Learning algorithm you just implemented is called the green version. 
	- In a blue green deployment, what you do is have the router send images to the old or the blue version and have that make decisions. And then when you want to switch over to the new version, what you would do is have the router stop sending images to the old one and suddenly switch over to the new version. 
	- So the way the blue green deployment is implemented is you would have an old prediction service may be running on some sort of service. You will then spin up a new prediction service, the green version, and you would have the router suddenly switch the traffic over from the old one to the new one. 
	- The advantage of a blue green deployment is that there's an easy way to enable rollback. If something goes wrong, you can just very quickly have the router go back reconfigure their router to send traffic back to the old or the blue version, assuming that you kept your blue version of the prediction service running.

<mark style="background: #ADCCFFA6;">Degrees of Automation</mark>
1. Human only system
2. Shadow mode
3. AI assistance
4. Partial automation: If the learning algorithm is confident in its prediction, we go with the algorithm's decision otherwise a human takes the decision.
5. Full automation