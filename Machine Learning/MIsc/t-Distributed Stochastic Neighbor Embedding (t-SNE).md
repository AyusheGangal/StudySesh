##### **<mark style="background: #D2B3FFA6;">About</mark>**
<mark style="background: #ABF7F7A6;">Important Features:</mark> 
- The goal is to take a set of points in a high-dimensional space and find a faithful representation of those points in a lower-dimensional space, typically the 2D plane. 
- The algorithm is non-linear and adapts to the underlying data, performing different transformations on different regions. Those differences can be a major source of confusion.
- A second feature of t-SNE is a tunable parameter, “<mark style="background: #ADCCFFA6;">perplexity</mark>” which says (loosely) how to balance attention between local and global aspects of your data. The parameter is, in a sense, a guess about the number of close neighbors each point has. The perplexity value has a complex effect on the resulting pictures. The original paper says, _“The performance of SNE is fairly robust to changes in the perplexity, and typical values are between 5 and 50.”_ 

<mark style="background: #ABF7F7A6;">Weird Drawback:</mark> 
- That’s not the end of the complications. The t-SNE algorithm doesn’t always produce similar output on successive runs, for example, and there are additional hyperparameters related to the optimization process.

One of the most widely used techniques for visualization is t-SNE, but its performance suffers with large datasets and using it correctly can be [challenging](https://distill.pub/2016/misread-tsne/).