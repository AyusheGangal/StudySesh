1. Takes high-dimensional data (3 or mor) and outputs a low-dimensional graph
2. Relatively fast with large datasets
3. Useful for idenifying similarities and outliers as similar samples tend to cluster together.

### <mark style="background: #D2B3FFA6;">UMAP for Sparse data</mark>
1. Essential libraries used
	- Sympy (has a lot of dependencies so will need to install a lot of extra libraries)
	- umap
	- scipy.sparse
	- basic: numpy, pandas, matplotlib
	
2. [A Mathematical Example](https://umap-learn.readthedocs.io/en/latest/sparse.html#a-mathematical-example): 
	We construct a sparse matrix consisting of prime numbers and the prime factors as columns. All this mathematics is performed using Sympy.

```Python
primes = list(sympy.primerange(2, 110000))
prime_to_column = {p:i for i, p in enumerate(primes)}
```

```Python
lil_matrix_rows = []
lil_matrix_data = []
for n in range(100000):
    prime_factors = sympy.primefactors(n)
    lil_matrix_rows.append([prime_to_column[p] for p in prime_factors])
    lil_matrix_data.append([1] * len(prime_factors))

factor_matrix = scipy.sparse.lil_matrix((len(lil_matrix_rows), len(primes)), dtype=np.float32)
factor_matrix.rows = np.array(lil_matrix_rows)
factor_matrix.data = np.array(lil_matrix_data)
```

Now we create a mapper object which fits this sparse data to the UMAP algorithm

```Python
%%time
mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True).fit(factor_matrix)

umap.plot.points(mapper, values=np.arange(100000), theme='viridis')
```

![[Screenshot 2022-11-28 at 3.54.06 PM.png]]

Generalizing the results for more unseen data
```Python
%%time
lil_matrix_rows = []
lil_matrix_data = []
for n in range(100000, 110000):
    prime_factors = sympy.primefactors(n)
    lil_matrix_rows.append([prime_to_column[p] for p in prime_factors])
    lil_matrix_data.append([1] * len(prime_factors))

new_data = scipy.sparse.lil_matrix((len(lil_matrix_rows), len(primes)), dtype=np.float32)
new_data.rows = np.array(lil_matrix_rows)
new_data.data = np.array(lil_matrix_data)
```

We use the `transform` method of our model to test this data
```Python
new_data_embedding = mapper.transform(new_data)
```

Plotting this using `matplotlib`
```Python
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
plt.scatter(new_data_embedding[:, 0], new_data_embedding[:, 1], s=0.1, c=np.arange(10000), cmap='viridis')
ax.set(xticks=[], yticks=[], facecolor='black');
```

![[Screenshot 2022-11-28 at 5.08.52 PM.png]]

On our dataset:
```Python
df_mapper = umap.UMAP(metric='cosine', random_state=42, low_memory=True).fit(df)
umap.plot.points(df_mapper, values=np.arange(len(df)), theme='viridis')
```

![[Screenshot 2022-11-28 at 5.12.17 PM.png]]