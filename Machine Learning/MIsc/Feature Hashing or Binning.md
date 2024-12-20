### <mark style="background: #D2B3FFA6;">Introduction</mark>
1. Feature hashing is a powerful technique for handling sparse, high-dimensional features in machine learning. It is fast, simple, memory-efficient, and well-suited to online learning scenarios. While an approximation, it has surprisingly low accuracy tradeoffs in many machine learning problems.
2. Feature hashing, or the hashing trick is a method for turning arbitrary features into a sparse binary vector. It can be extremely efficient by having a standalone hash function that requires no pre-built dictionary of possible categories to function.

### <mark style="background: #D2B3FFA6;">Dealing With Sparse, High-Dimensional Features</mark>
1. Many domains have very high feature dimension (the number of features or variables in the dataset), while some feature domains are naturally dense (for example, images and video).
2. Here, we're concerned with sparse, high-cardinality feature domains such as those found in online advertising, e-commerce, social networks, text, and natural language processing.
3. <mark style="background: #ABF7F7A6;">By high-cardinality features, we mean that there are many unique values that the feature can take on. Feature cardinality in these domains can be of the order of millions or hundreds of millions</mark> (for example, the unique user IDs, product IDs, and search keywords in online advertising). Feature sparsity means that for anyone training example the number of active features is extremely low relative to the feature dimension — this ratio is often less than 1%.

##### <mark style="background: #ADCCFFA6;">Challenge</mark>
This creates a challenge at scale because even simple models can become very large, making them consume more memory and resources during training and when making predictions.

### <mark style="background: #D2B3FFA6;">The "Hashing Trick"</mark>
- <mark style="background: #ADCCFFA6;">Core-idea:</mark>  is relatively simple. Instead of maintaining a one-to-one mapping of categorical feature values to locations in the feature vector, we use a hash function to determine the feature's location in a vector of lower dimension.
- <mark style="background: #ADCCFFA6;">Example 1:</mark> For instance, to determine the vector index for a given feature value, we apply a hash function to that feature value (say, `city=Boston`) to generate a hash value. We then effectively project that hash value into the lower size of our new feature vector to generate the vector index for that feature value (using a modulo operation).
- <mark style="background: #ADCCFFA6;">Example 2:</mark> The same idea applies to dealing with text — the feature indices are computed by hashing each word in the text, while the only difference is that the feature values represent counts of the occurrence of each word (as in the bag-of-words encoding) instead of binary indicators (as in the one-hot encoding). 
- <mark style="background: #ABF7F7A6;">In order to ensure the hashed feature indices are evenly spread throughout the vector, the hashed dimension is typically chosen to be a power of 2</mark> (where the power is referred to as the number of hash bits).

##### <mark style="background: #D2B3FFA6;">Possible Collisions</mark>
- When using hashing combined with the modulo operation, we may encounter hash collisions this occurs when two feature values end up being hashed to the same index in the vector.
- It turns out that in <mark style="background: #ABF7F7A6;">practice for sparse feature domains, this has very little impact on the performance characteristics of our machine learning algorithm</mark>.
- The reasoning behind this is that for very sparse features, relatively few tend to be informative, and in aggregate we hope that any hash collisions will impact less informative features. In a sense, hash collisions add a type of "noise" to our data — and our machine learning algorithms are designed to pick out the signal from the noise, so their predictive accuracy doesn't suffer too much.
- Using feature hashing, we can gain significant advantages in memory usage since we can bound the size of our feature vectors and we don't need to store the exact one-to-one mapping of features to indices that are required with other encoding schemes. In addition, we can do this very fast, with little loss of accuracy, while preserving sparsity. <mark style="background: #ABF7F7A6;">Because of these properties, feature hashing is well-suited to online learning scenarios, systems with limited resources, or when speed and simplicity are important.</mark>

 **<mark style="background: #ADCCFFA6;">Major Problem:</mark>** In addition to the potential for hash collisions mentioned above, <mark style="background: #ABF7F7A6;">we lose the ability to perform the inverse mapping from feature indices back to feature values, precisely because we don't generate or store the explicit feature mapping during feature extraction.</mark> This can be problematic for inspecting features (for example, we can compute which features are most informative for our model, but we are unable to map those back to the raw feature values in our data for interpretation).

