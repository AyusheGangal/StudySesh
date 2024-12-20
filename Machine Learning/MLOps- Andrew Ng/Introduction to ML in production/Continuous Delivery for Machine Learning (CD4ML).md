Notes from the [article](https://martinfowler.com/articles/cd4ml.html)
- According to the paper  [_"Hidden Technical Debt in Machine Learning Systems"_,](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems) published by Sculley et al. in 2015, in real-world Machine Learning (ML) systems only a small fraction is comprised of actual ML.
- Also talks about many sources of technical debt that can accumulate in such systems, some of which are related to data dependencies, model complexity, reproducibility, testing, monitoring, and dealing with changes in the external world.

##### **<mark style="background: #D2B3FFA6;">Continuous Delivery</mark>**
- An Approach to bring automation, quality, and discipline to create a reliable and repeatable process to release software into production.
- In the seminar book _"Continuous Delivery"_, by Jez Humble and David Farley state that:

	>"Continuous Delivery is the ability to get changes of all types — including new features, configuration changes, bug fixes, and experiments — into production, or into the hands of users, safely and quickly in a sustainable way".
	>
	>- Jez Humble and Dave Farley

- Besides the code, changes to ML models and the data used to train them are another type of change that needs to be managed and baked into the software delivery process![[Screenshot 2023-01-11 at 2.07.52 PM.png|500]]


#### **<mark style="background: #D2B3FFA6;">Continuous Delivery for Machine Learning (CD4ML)</mark>
###### <mark style="background: #ABF7F7A6;">Definition</mark>
- It is a software engineering approach in which a cross-functional team produces machine learning applications based on code, data, and models in small and safe increments that can be reproduced and reliably released at any time, in short adaptation cycles.
- Breaking the definition down:
	- **<mark style="background: #ADCCFFA6;">Software engineering approach:</mark>** It enables teams to efficiently produce high quality software.
	- **<mark style="background: #ADCCFFA6;">Cross-functional team:</mark>** Experts with different skill sets and workflows across data engineering, data science, machine learning engineering, development, operations, and other knowledge areas are working together in a collaborative way emphasizing the skills and strengths of each team member.
	- **<mark style="background: #ADCCFFA6;">Producing software based on code, data, and machine learning models:</mark>** All artifacts of the ML software production process require different tools and workflows that must be versioned and managed accordingly.
	- **<mark style="background: #ADCCFFA6;">Small and safe increments:</mark>** The release of the software artifacts is divided into small increments, which allows visibility and control around the levels of variance of its outcomes, adding safety into the process.
	- **<mark style="background: #ADCCFFA6;">Reproducible and reliable software release:</mark>** While the model outputs can be non-deterministic and hard to reproduce, the process of releasing ML software into production is reliable and reproducible, leveraging automation as much as possible.
	- **<mark style="background: #ADCCFFA6;">Software release at any time:</mark>** It is important that the ML software could be delivered into production at any time. Even if organizations do not want to deliver software all the time, it should always be in a releasable state. This makes the decision about when to release it a business decision rather than a technical one.
	- **<mark style="background: #ADCCFFA6;">Short adaptation cycles:</mark>** Short cycles means development cycles are in the order of days or even hours, not weeks, months or even years. Automation of the process with quality built in is key to achieve this. This creates a feedback loop that allows you to adapt your models by learning from its behavior in production.

###### <mark style="background: #ABF7F7A6;">Challenges</mark>
1. <mark style="background: #ADCCFFA6;">Organizational structure:</mark> different teams might own different parts of the process, and there is a hand over — or usually, "throw over the wall" — without clear expectations of how to cross these boundaries
	- For example, 
		- Data Engineers might be building pipelines to make data accessible.
		- Data Scientists are worried about building and improving the ML model. 
		- Then Machine Learning Engineers or developers will have to worry about how to integrate that model and release it to production.![[Screenshot 2023-01-11 at 3.21.21 PM.png]]
	- This leads to delays and friction. <mark style="background: #FFB86CA6;">A common symptom is having models that only work in a lab environment and never leave the proof-of-concept phase.</mark> Or if they make it to production, in a manual ad-hoc way, they become stale and hard to update.
	
2.  <mark style="background: #ADCCFFA6;">Technical:</mark> 
	- How to make the process reproducible and auditable. 
	- Because these teams use different tools and follow different workflows, it becomes hard to automate it end-to-end. 
	- There are more artifacts to be managed beyond the code, and versioning them is not straightforward. 
	- Some of them can be really large, requiring more sophisticated tools to store and retrieve them efficiently.

>[!Up Next]
>
There are a number of [[Technical Components of CD4ML]]. 
