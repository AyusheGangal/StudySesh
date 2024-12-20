### <mark style="background: #D2B3FFA6;">Monitoring Dashboard</mark>
- The best practice to monitor an ML system is to use a dashboard to track how its doing over time. 
- Can use different dashboards to keep track of different metrics, eg, Server Load, Non-null outputs fraction, fraction of missing inout values

#### <mark style="background: #D2B3FFA6;">Best practices:</mark>
1. Brainstorm the things that could go wrong
2. Brainstorm a few statistics/metrics that will detect the problem
3. it's okay to start off with a lot of different metrics and monitor a relatively large set and then gradually remove the ones that you find over time not to be particularly useful. 
4. After you've chosen a set of metrics, a common practice is to set thresholds for alarms. Eg, if server load ever goes beyond 0.9, you'll get a notification if there's a problem.
5. Adapt metrics and threshold over time.

###### <mark style="background: #ABF7F7A6;">Examples of metrics to use:</mark>
1. <mark style="background: #ADCCFFA6;">Software metrics: </mark>Memory, compute, latency, throughput, server load
2. <mark style="background: #ADCCFFA6;">Input metrics:</mark> metrics that measure if your input distribution x change. Eg, Avg input length, Avg input volume, Num of missing values, Avg image brightness
3. <mark style="background: #ADCCFFA6;">Output metrics: </mark># times return "" (null), # times user redoes search, # times user switches to typing, Click-through rate (CTR)

Just as ML modeling is iterative, so is deployment.
![[Screenshot 2023-01-09 at 3.32.39 PM.png|500]]

<mark style="background: #ABF7F7A6;">Model maintenance:</mark>
1. Manual retraining (far more common)
2. Automatic retraining

### <mark style="background: #D2B3FFA6;">Pipeline Monitoring</mark>
Metrics which keep track of concept drift or data drift or both, and in multiple stages of the pipeline.
- Software metrics: for each of the components of the pipeline or for the overall pipeline as a whole.
- Input metrics
- Output metrics

How quickly do the metrics change? - extremely problem dependent
- user data generally has slower drift.
- Enterprise data (B2B applications) can shift fast

