<mark style="background: #ADCCFFA6;">What to track?</mark>
1. Algorithm/ Code versioning
2. Dataset used
3. Hyper-parameters
4. Save the results somewhere

<mark style="background: #ADCCFFA6;">Tracking tools:</mark>
1. Text files (okay for small experiments)
2. Spread sheets
3. Experiment tracking systems: Weights & Biases, Comet, MLflow, Sage Maker Studio

<mark style="background: #ADCCFFA6;">Desirable Features:</mark>
1. Information needed to replicate results. Note: if the learning algorithm pulls data off the internet as the data on the internet can change, that can decrease replicability .
2. Experiment results, ideally with summary metrics/ analysis.
3. Resource monitoring, visualization, model error analysis.


### <mark style="background: #D2B3FFA6;">From Big Data to Good Data</mark>
Try to ensure consistently high-quality data in all phases of the ML project lifecycle.
Good Data:
1. Covers important cases (good coverage of inputs x)
2. Is defined consistently (definition of labels y in unambiguous)
3. Has timely feedback from production data (distribution covers data drift and concept drift)
4. Is sized appropriately 