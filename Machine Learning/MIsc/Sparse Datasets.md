1. Sparse data is a variable in which the cells do not contain actual data within data analysis. [Sparse data is empty or has a zero value](https://sisudata.com/blog/neural-networks-sparse-data-graph-coloring). 
2. Sparse data is different from missing data because sparse data shows up as empty or zero while missing data doesn’t show what some or any of the values are.

##### **<mark style="background: #D2B3FFA6;">Types of sparsity:</mark>**
-  **Controlled sparsity** happens when there is a range of values for multiple dimensions that have no value
-   **Random sparsity** occurs when there is sparse data scattered randomly throughout the datasets

Many companies use a sparse matrix, a matrix that contains most elements of zero. If you’re using a sparse matrix, it is highly recommended that you use specific data structures to keep your data storage efficient.

> Handling a sparse matrix as a dense one is frequently inefficient, making excessive use of memory. When working with sparse matrices it is recommended to use dedicated data structures for efficient storage and processing. 

Sparse datasets with high zero values can cause problems like over-fitting in the machine learning models and several other problems. That is why dealing with sparse data is one of the most hectic processes in machine learning.

Most of the time, sparsity in the dataset is not a good fit for the machine learning problems in it should be handled properly. Still, sparsity in the dataset is good in some cases as it reduces the memory footprint of regular networks to fit mobile devices and shortens training time for ever-growing networks in deep learning.

##### **<mark style="background: #D2B3FFA6;">Problems caused by Sparsity</mark>**
1. **<mark style="background: #ADCCFFA6;">Over-fitting</mark>:** if there are <mark style="background: #ABF7F7A6;">too many features included in the training data, then while training a model, the model with tend to follow every step of the training data</mark>, results in higher accuracy in training data and lower performance in the testing dataset.
2. **<mark style="background: #ADCCFFA6;">Avoiding Important Data</mark>:** Some machine-learning algorithms avoid the importance of sparse data and only tend to train and fit on the dense dataset. They do not tend to fit on sparse datasets. The avoided sparse data can also have some training power and useful information, which the algorithm neglects. So it is not always a better approach to deal with sparse datasets.
3. **<mark style="background: #ADCCFFA6;">Space Complexity</mark>**: If the dataset has a sparse feature, it will take more space to store than dense data; hence, the space complexity will increase. Due to this, higher computational power will be needed to work with this type of data.
4. **<mark style="background: #ADCCFFA6;">Time Complexity</mark>**: If the dataset is sparse, then training the model will take more time to train compared to the dense dataset on the data as the size of the dataset is also higher than the dense dataset.
5. **<mark style="background: #ADCCFFA6;">Change in Behavior of the algorithms</mark>**: Some of the algorithms might perform badly or low on sparse datasets. Some algorithms tend to perform badly while training them on sparse datasets. <mark style="background: #ABF7F7A6;">Logistic Regression is one of the algorithms which shows flawed behavior in the best fit line while training it on a space dataset. Random Forest also tend to perform badly when applied on sparse datasets.</mark>

##### **<mark style="background: #D2B3FFA6;">How to deal with Sparsity</mark>**
1. Convert from Sparse to Dense
	1. Using PCA ([[Principle Component Analysis]])
	2. Apply [[Feature Hashing or Binning]]
	3. Apply Feature Selection and Feature Extraction
	4. Use [[t-Distributed Stochastic Neighbor Embedding (t-SNE)]]
	
2. Remove Features from the model
	1. drop columns using pandas - drift time (if need be)

3. Use methods which are not affected by the Sparse data
	1. [[Entropy-weighted k means algorithm]]
	2. [[UMAP-Uniform Manifold Approximation and Projection for Dimension Reduction]]
