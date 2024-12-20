![[Screenshot 2023-01-12 at 11.51.07 AM.png]]

##### <mark style="background: #D2B3FFA6;">Continuous Delivery for Machine Learning end-to-end process</mark>
- At the base, we need an easy way to manage, discover, access, and version our data. We then automate the model building and training process to make it reproducible. 
- This allows us to experiment and train multiple models, which brings a need to measure and track those experiments. 
- Once we find a suitable model, we can decide how it will be production-ized and served. Because the model is evolving, we must ensure that it won't break any contract with its consumers, therefore we need to test it before deploying to production. 
- Once in production, we can use the monitoring and observability infrastructure to gather new data that can be analyzed and used to create new training data sets, closing the feedback loop of continuous improvement.
- A Continuous Delivery orchestration tool coordinates the end-to-end CD4ML process, provisions the desired infrastructure on-demand, and governs how models and applications are deployed to production.