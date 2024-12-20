###### <mark style="background: #ABF7F7A6;">Technical Components of CD4ML</mark>
- **<mark style="background: #ADCCFFA6;">Discoverable and Accessible Data</mark>**
	- [[Data Lake Architecture]]
	- [[Data Mesh Architecture]]
	- <mark style="background: #FFB86CA6;">Data Pipelines:</mark> 
		- The process that takes input data through a series of transformation stages, producing data as output. Both the input and output data can be fetched and stored in different locations, such as a database, a stream, a file, etc. 
		- The transformation stages are usually defined in code, although some ETL tools allow you to represent them in a graphical form. They can be executed either as a batch job, or as a long-running streaming application.
		- For the purposes of CD4ML, we treat a data pipeline as an artifact, which can be version controlled, tested, and deployed to a target execution environment.
	
- **<mark style="background: #ADCCFFA6;">Reproducible Model Training</mark>**
	- <mark style="background: #FFB86CA6;">Machine Learning Pipelines/ Model Training Pipelines:</mark>
		- The process that takes data and code as input, and produces a trained ML model as the output. 
		- This process usually involves data cleaning and pre-processing, feature engineering, model and algorithm selection, model optimization and evaluation.
		- While developing this process encompasses a major part of a Data Scientist's workflow for the purposes of CD4ML, we treat the ML pipeline as the final automated implementation of the chosen model training process.
		
	- Once the data is available, we move into the iterative Data Science workflow of model building. 
		- This usually involves splitting the data into a training set and a validation set
		- trying different combinations of algorithms
		- and tuning their parameters and hyper-parameters. 
		- That produces a model that can be evaluated against the validation set, to assess the quality of its predictions. The step-by-step of this model training process becomes the machine learning pipeline.
		
	- ![[Screenshot 2023-01-11 at 10.20.53 PM.png|500]]
	- Example of a structured ML pipeline for sales forecasting problem highlighting the different source code, data, and model components. 
	- The input data, the intermediate training and validation data sets, and the output model can potentially be large files, which we don't want to store in the source control repository. Also, the stages of the pipeline are usually in constant change, which makes it hard to reproduce them outside of the Data Scientist's local environment.
	

- **<mark style="background: #ADCCFFA6;">Model Serving</mark>**
	- Once a suitable model is found, we need to decide how it will be served and used in production. We have seen a few patterns to achieve that:
		- **<mark style="background: #FFB86CA6;">Embedded model:</mark>** this is the simpler approach, where you treat the model artifact as a dependency that is built and packaged within the consuming application. From this point forward, you can treat the application artifact and version as being a combination of the application code and the chosen model.
		
		- **<mark style="background: #FFB86CA6;">Model deployed as a separate service:</mark>** in this approach, the model is wrapped in a service that can be deployed independently of the consuming applications. This allows updates to the model to be released independently, but it can also introduce latency at inference time, as there will be some sort of remote invocation required for each prediction.
		
		- **<mark style="background: #FFB86CA6;">Model published as data:</mark>** in this approach, the model is also treated and published independently, but the consuming application will ingest it as data at runtime. We have seen this used in streaming/real-time scenarios where the application can subscribe to events that are published whenever a new model version is released, and ingest them into memory while continuing to predict using the previous version. Software release patterns such as Blue Green Deployment or Canary Releases can also be applied in this scenario.
	
- **<mark style="background: #ADCCFFA6;">Testing and Quality in Machine Learning</mark>** 
	There are different types of testing that can be introduced in the ML workflow. There are many types of automated tests that can add value and improve the overall quality of an ML system:
	- **<mark style="background: #FFB86CA6;">Validating data:</mark>** 
		- We can add tests to validate input data against the expected schema, or to validate our assumptions about its valid values — e.g. they fall within expected ranges, or are not null. 
		- For engineered features, we can write unit tests to check they are calculated correctly — e.g. numeric features are scaled or normalized, one-hot encoded vectors contain all zeroes and a single 1, or missing values are replaced appropriately.
		
	- **<mark style="background: #FFB86CA6;">Validating component integration:</mark>** 
		- We can use a similar approach to testing the integration between different services, using <mark style="background: #FFF3A3A6;">Contract Tests</mark> to validate that the expected model interface is compatible with the consuming application. 
		- Another type of testing that is relevant when your model is productionized in a different format, is to make sure that the exported model still produces the same results. This can be achieved by running the original and the productionized models against the same validation dataset, and comparing the results are the same.
		
	- **<mark style="background: #FFB86CA6;">Validating the model quality:</mark>** 
		- While ML model performance is non-deterministic, Data Scientists usually collect and monitor a number of metrics to evaluate a model's performance, such as error rates, accuracy, AUC, ROC, confusion matrix, precision, recall, etc. 
		- They are also useful during parameter and hyper-parameter optimization. As a simple quality gate, we can use these metrics to introduce [[Threshold Tests]] or [[ratcheting]] in our pipeline, to ensure that new models don't degrade against a known performance baseline.
		
	- **<mark style="background: #FFB86CA6;">Validating model bias and fairness:</mark>**
		- while we might get good performance on the overall test and validation datasets, it is also important to check how the model performs against baselines for specific data slices. 
		- For instance, you might have inherent bias in the training data where there are many more data points for a given value of a feature (e.g. race, gender, or region) compared to the actual distribution in the real world, so it's important to check performance across different slices of the data. A tool like [Facets](https://pair-code.github.io/facets/) can help you visualize those slices and the distribution of values across the features in your datasets.
	
	![[Screenshot 2023-01-12 at 10.15.35 AM.png|]]
	An example of how to combine different test pyramids for data, model, and code in CD4ML
	
- **<mark style="background: #ADCCFFA6;">Experiment Tracking</mark>** 
	- In order to support this governance process, it is important to capture and display information that will allow humans to decide if and which model should be promoted to production. 
	- As the Data Science process is very research-centric, it is common that you will have multiple experiments being tried in parallel, and many of them might not ever make it to production.
	- This experimentation approach during the research phase is different than a more traditional software development process, as we expect that the code for many of these experiments will be thrown away, and only a few of them will be deemed worthy of making it to production. For that reason, we will need to define an approach to track them.
	- To support this experimentation process, it is also important to highlight the benefits of having <mark style="background: #FFF3A3A6;">elastic infrastructure</mark>, as you might need multiple environments to be available — and sometimes with specialized hardware — for training. <mark style="background: #FFF3A3A6;">Cloud-based infrastructure</mark> is a natural fit for this, and many of the public cloud providers are building services and solutions to support various aspects of this process.
	
- **<mark style="background: #ADCCFFA6;">Model Deployment</mark>** 
	- **<mark style="background: #FFB86CA6;">Multiple Models:</mark>** Sometimes we might have more than one model performing the same task. For example, we could train a model to predict demand for each product. In that case, deploying the models as a separate service might be better for consuming applications to get predictions with a single API call.
	
	- **<mark style="background: #FFB86CA6;">Shadow Models:</mark>** This pattern is useful when considering the replacement of a model in production. We can deploy the new model side-by-side with the current one, as a _shadow model_, and send the same production traffic to gather data on how the shadow model performs before promoting it.
	
	- **<mark style="background: #FFB86CA6;">Competing Models:</mark>** 
		- A slightly more complex scenario is when you are trying multiple versions of the model in production — like an A/B test — to find out which one is better. 
		- The added complexity here comes from the infrastructure and routing rules required to ensure the traffic is being redirected to the right models, and that you need to gather enough data to make statistically significant decisions, which can take some time. 
		- Another popular approach for evaluating multiple competing models is <mark style="background: #FFF3A3A6;">Multi-Armed Bandits</mark>, which also requires you to define a way to calculate and monitor the reward associated with using each model. Applying this to ML is an active area of research, and we are starting to see some tools and services appear, such as [Seldon core](https://docs.seldon.io/projects/seldon-core/en/latest/analytics/routers.html) and [Azure Personalizer](https://docs.microsoft.com/en-us/azure/cognitive-services/personalizer/) .
		
	-  **<mark style="background: #FFB86CA6;">Online Learning Models:</mark>**  Unlike the models we discussed so far, that are trained offline and used online to serve predictions, [online learning](http://www.mit.edu/~rakhlin/6.883/) models use algorithms and techniques that can continuously improve its performance with the arrival of new data. They are constantly learning in production. This poses extra complexities, as versioning the model as a static artifact won't yield the same results if it is not fed the same data. You will need to version not only the training data, but also the production data that will impact the model's performance.
	
- **<mark style="background: #ADCCFFA6;">Continuous Delivery Orchestration</mark>** 
	- **<mark style="background: #FFB86CA6;">Deployment Pipelines:</mark>** A deployment pipeline automates the process for getting software from version control into production, including all the stages, approvals, testing, and deployment to different environments. In CD4ML, we can model automated and manual ML governance stages into our deployment pipeline, to help detect model bias, fairness, or to introduce explainability for humans to decide if the model should further progress towards production or not.
	
	- With all of the main building blocks in place, there is a need to tie everything together, and this is where Continuous Delivery orchestration tools come into place.
	- In CD4ML, we have extra requirements to orchestrate: the provisioning of infrastructure and the execution of the Machine Learning Pipelines to train and capture metrics from multiple model experiments; the build, test, and deployment process for our Data Pipelines; the different types of testing and validation to decide which models to promote; the provisioning of infrastructure and deployment of our models to production.
	
- **<mark style="background: #ADCCFFA6;">Model Monitoring and Observability</mark>** 
	- Once the model is live, we need to understand how it performs in production and close the data feedback loop.
	- Here we can reuse all the monitoring and observability infrastructure that might already be in place for the applications and services. 
	- We can understand how the model is behaving by observing:
		- <mark style="background: #FFB86CA6;">Model Inputs:</mark> What data is being fed to the models, giving visibility into any training-serving skew.
		- <mark style="background: #FFB86CA6;">Model Outputs:</mark> What predictions and recommendations are the models making from these inputs, to understand how the model is performing with real data.
		- <mark style="background: #FFB86CA6;">Model interpretability outputs:</mark> Metrics such as model coefficients, [ELI5](https://eli5.readthedocs.io/en/latest), or [LIME](https://arxiv.org/abs/1602.04938) outputs that allow further investigation to understand how the models are making predictions to identify potential overfit or bias that was not found during training.
		- <mark style="background: #FFB86CA6;">Model outputs and decisions:</mark> What predictions our models are making given the production input data, and also which decisions are being made with those predictions. Sometimes the application might choose to ignore the model and make a decision based on pre-defined rules (or to avoid future bias).
		- <mark style="background: #FFB86CA6;">User action and rewards:</mark> Based on further user action, we can capture reward metrics to understand if the model is having the desired effect. For example, if we display product recommendations, we can track when the user decides to purchase the recommended product as a reward.
		- <mark style="background: #FFB86CA6;">Model fairness:</mark> Analyzing input data and output predictions against known features that could bias, such as race, gender, age, income groups, etc.

>[!Up Next]
>[[The End-to-End CD4ML Process]]

