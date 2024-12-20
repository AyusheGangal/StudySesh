source: https://towardsdatascience.com/logarithms-exponents-in-complexity-analysis-b8071979e847

**_log(n)_** increases only by a tiny amount as **N** increases. When **N** doubles, **_log(n)_** only increases by 1. And so this is why, if we tie this back to complexity analysis when we have an algorithm with time complexity of _log(n)_, that is incredibly good because that means as the input increases/doubles, the number of elementary operations that we’re performing in the algorithm only increases by one.

**Logarithmic time complexity** **_log(n)_**: Represented in Big O notation as **O(log n)**, when an algorithm has O(log n) running time, it means that as the input size grows, the number of operations grows very slowly. **Example:** binary search.

![[Screenshot 2022-10-31 at 6.01.35 PM.png]]