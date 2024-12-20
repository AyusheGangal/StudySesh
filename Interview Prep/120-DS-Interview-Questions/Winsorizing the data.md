[[Winsorization]] is a way to minimize the influence of outliers in the data by either
-   Assigning the outlier a lower weight,
-   Changing the value so that it is close to other values in the set.

The data points are _modified_, not trimmed/removed.

[Statistics](https://www.statisticshowto.com/statistic/) such as the [mean](https://www.statisticshowto.com/mean/) and [variance](https://www.statisticshowto.com/probability-and-statistics/variance/) are very susceptible to outliers; Winsorization can be an effective way to deal with this problem, improve statistical [efficiency](https://www.statisticshowto.com/efficient-estimator-efficiency/) and increase the [robustness](https://www.statisticshowto.com/robust-statistics/) of statistical inferences. 

1. The downside is that [bias](https://www.statisticshowto.com/what-is-bias/) is introduced into your results, although the bias is a lot less than if you had simply deleted the data point. 
2. The alternative is to keep the data point as-is, but that may not be the best choice as it could dramatically skew your results. 
3. Either way, you should always have a good justification for Winsorizing your data; Never run the procedure arbitrarily in the hopes of getting more [significant](https://www.statisticshowto.com/what-is-statistical-significance/)results.

