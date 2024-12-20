### <mark style="background: #D2B3FFA6;">Lifecycle steps for the "Speech recognition" example</mark>

1. **<mark style="background: #ABF7F7A6;">Scoping</mark>**
	- Defining the project involves:
		- Make a decision to work on speech recognition for voice search
		- Decide on key metrics: it is extremely problem-dependent. Almost all applications have a unique set of goals and metrics.
			- Accuracy, latency, throughput (how many queries/second)
		- Estimate the resources needed
			- Time, Compute, budget, timeline
	
2. **<mark style="background: #ABF7F7A6;">Data</mark>**
	- Define data: 
		- challenge: Inconsistent Data
		- How much silence before/after each clip?
		- How to perform volume normalization?
	- Label and organize data
	
3. **<mark style="background: #ABF7F7A6;">Modeling</mark>**
	- Three key inputs that go into training an ML model:
		- Code (Algorithm/model)
		- Hyper parameters
		- Data
	-  For Research/ Academia, we keep the data fixed and vary the Code and the hyper parameters, whereas for Product Team, we keep the Code fixed and vary the data and hyper parameters.
	- Select and train model
	- Use error analysis to improve where your model falls short. This is a good trick to obtain a higher accuracy model. 
	
4. **<mark style="background: #ABF7F7A6;">Deployment</mark>** ![[Screenshot 2023-01-09 at 10.52.03 AM.png]]
	- **<mark style="background: #ADCCFFA6;">Workflow:</mark>**
		- We have a mobile phone, which is the edge device with the software running locally on the phone.
		- The software taps into the microphone to record what someone is saying. 
		- We use a VAD (voice activity detection) module which enables the smartphone to select out just the audio of someone speaking which we can then send to our prediction server which is the cloud.
		- The prediction server returns both the transcript, so the user sees what the system thinks the user said, and the search results.
		- The transcript and the search results are displayed in the Front end code.
	- Challenge in deployment: Concept drift/ Data drift, when the data distribution changes.
	