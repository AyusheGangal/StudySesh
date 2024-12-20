Two major categories of challenges in deploying an ML model:
1. <mark style="background: #D2B3FFA6;">Machine learning/ Statistical Issues</mark>
- <mark style="background: #ABF7F7A6;">Concept Drift and Data Drift</mark>
	- <mark style="background: #ADCCFFA6;">For speech recognition example</mark>
		- Training set: Purchased data, historic data with transcripts ($X\rightarrow Y$)
		- Test set: Data from a few months ago
	- <mark style="background: #ADCCFFA6;">Data Drift: </mark>
		- Aka data drift, feature drift, population or covariate shift.
		- How has the data changed?- data distribution, language changed, microphone changed so the audio sounds different, then the performance of a speech recognition system can degrade.
		- "The distribution of the variables is meaningfully different. As a result, the trained model is not relevant for this new data."
		- It is important to know how the data changes and if you need to update the learning algorithm as a result.
		- Sometimes data changes gradually, and sometimes it changes suddenly like a sudden shock. 
	- <mark style="background: #ADCCFFA6;">Concept Drift:</mark> 
		- When the desired mapping (relationship) from $X\rightarrow Y$ changes, ie., when the patterns the model learned no longer hold.
		- Types: 
			- Gradual or incremental 
			- Sudden or drastic
	
- <mark style="background: #ADCCFFA6;">Training-serving skew:</mark>
	- Used inter-changeably with data drift but isn't exactly drift as there is no change during the production use of the model.
	- Training-serving skew is more of a mismatch. It reveals at the first attempt to apply the model to the real data.
	- It often happens when you train a model on an artificially constructed or cleaned dataset. This data does not necessarily represent the real world, or does this incompletely.
	- In most cases of training-serving skew, the model development has to continue.
	
1. <mark style="background: #D2B3FFA6;">Software engineering issues</mark>
- For a "Prediction Service" which takes input X and gives output Y
	- <mark style="background: #ABF7F7A6;">checklist of questions: </mark>
		- Realtime or Batch
		- Cloud vs Edge/Browser
		- Compute resources (CPU/GPU/memory)
		- Latency, throughput (QPS)
		- Logging
		- Security and Privacy

First Deployment Vs Maintenance


