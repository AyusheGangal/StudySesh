## <mark style="background: #D2B3FFA6;">Hierarchical indexing</mark>

Link: https://pandas.pydata.org/docs/user_guide/advanced.html
1. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like `Series` (1d) and `DataFrame` (2d).

Multi-indexed or hierarchical data refers to data that has multiple levels of indexing or grouping, typically organized in a tree-like structure. Each level of the index represents a different dimension or aspect of the data, allowing for more complex and flexible analyses.

In Python, Pandas is a popular library for working with hierarchical data, and it provides several data structures for representing multi-indexed data. The most common one is the MultiIndex, which allows for indexing and selecting data based on multiple levels of labels.

For example, suppose we have a dataset of sales data with the following columns: year, quarter, region, and sales. We could use a MultiIndex to represent this data as a hierarchy, with the first level of the index representing the year, the second level representing the quarter, and the third level representing the region.

```python
import pandas as pd

sales_data = pd.read_csv('sales_data.csv')
sales_data = sales_data.set_index(['year', 'quarter', 'region'])
```

Once the data is indexed, we can easily select subsets of the data based on any combination of the index levels, using the `.loc` accessor

```python
q3_2022_east_sales = sales_data.loc[(2022, 'Q3', 'East'), 'sales']
```

We can also perform group-level operations on the data, such as computing the mean sales for each quarter and region:

```python
quarterly_sales = sales_data.groupby(level=['quarter', 'region'])['sales'].mean()
```

Hierarchical data structures like the MultiIndex can be extremely useful for analyzing complex datasets with multiple dimensions, and can provide a powerful tool for data exploration and visualization. Multi-indexed or hierarchical data structures can be useful in a variety of data analysis tasks. Here are some common use cases:

1.  Organizing and summarizing data: Multi-indexed data structures can be used to group and summarize data along multiple dimensions. For example, you could use a multi-index to group sales data by year, quarter, and region, and then compute summary statistics for each group.
    
2.  Time series analysis: Time series data often has a natural hierarchy, such as year, month, and day. Multi-indexed data structures can be used to organize and analyze time series data, making it easy to compute rolling averages, seasonality, and other time-based metrics.
    
3.  Hierarchical modeling: In statistical modeling, it's often useful to include hierarchical structures in your models. Multi-indexed data structures can be used to represent the levels of the hierarchy, making it easy to fit models that account for both within-group and between-group variation.
    
4.  Data visualization: Hierarchical data structures can be visualized using tree diagrams, heatmaps, and other graphical representations. This can help you gain insights into the structure of your data and identify patterns and trends.
    
5.  Machine learning: Multi-indexed data structures can be used in machine learning tasks such as classification and regression. For example, you could use a multi-index to represent the inputs and outputs of a neural network, or to organize training data for a decision tree classifier.