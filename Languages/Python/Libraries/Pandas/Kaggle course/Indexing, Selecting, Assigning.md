The indexing operator and attribute selection are nice because they work just like they do in the rest of the Python ecosystem. As a novice, this makes them easy to pick up and use. However, pandas has its own accessor operators, `loc` and `iloc`. For more advanced operations, these are the ones you're supposed to be using.

### Index-based selection
Pandas indexing works in one of two paradigms. The first is **index-based selection**: selecting data based on its numerical position in the data. `iloc` follows this paradigm.

To select the first row of data in a DataFrame, we may use the following:
``` python
reviews.iloc[0]
```

Both `loc` and `iloc` are row-first, column-second. This is the opposite of what we do in native Python, which is column-first, row-second.

This means that it's marginally easier to retrieve rows, and marginally harder to get retrieve columns. To get a column with `iloc`, we can do the following:
```Python
reviews.iloc[:, 0]
```

On its own, the `:` operator, which also comes from native Python, means "everything". When combined with other selectors, however, it can be used to indicate a range of values. For example, to select the `country` column from just the first, second, and third row, we would do:
```Python
reviews.iloc[:3, 0]
```

It's also possible to pass a list:
```Python
reviews.iloc[[0, 1, 2], 0]
```

### Label-based selection
The second paradigm for attribute selection is the one followed by the `loc` operator: **label-based selection**. In this paradigm, it's the data index value, not its position, which matters.

For example, to get the first entry in `reviews`, we would now do the following:
```Python
reviews.loc[0, 'country']
```

`iloc` is conceptually simpler than `loc` because it ignores the dataset's indices. When we use `iloc` we treat the dataset like a big matrix (a list of lists), one that we have to index into by position. `loc`, by contrast, uses the information in the indices to do its work. Since your dataset usually has meaningful indices, it's usually easier to do things using `loc` instead. For example, here's one operation that's much easier using `loc`:
```Python
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
```

### Choosing between `loc` and `iloc`
When choosing or transitioning between `loc` and `iloc`, there is one "gotcha" worth keeping in mind, which is that the two methods use slightly different indexing schemes.

`iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So `0:10` will select entries `0,...,9`. `loc`, meanwhile, indexes inclusively. So `0:10` will select entries `0,...,10`.

Why the change? Remember that loc can index any stdlib type: strings, for example. If we have a DataFrame with index values `Apples, ..., Potatoes, ...`, and we want to select "all the alphabetical fruit choices between Apples and Potatoes", then it's a lot more convenient to index `df.loc['Apples':'Potatoes']` than it is to index something like `df.loc['Apples', 'Potatoet']` (`t` coming after `s` in the alphabet).

This is particularly confusing when the DataFrame index is a simple numerical list, e.g. `0,...,1000`. In this case `df.iloc[0:1000]` will return 1000 entries, while `df.loc[0:1000]` return 1001 of them! To get 1000 elements using `loc`, you will need to go one lower and ask for `df.loc[0:999]`.

Otherwise, the semantics of using `loc` are the same as those for `iloc`.

# Manipulating the index
Label-based selection derives its power from the labels in the index. Critically, the index we use is not immutable. We can manipulate the index in any way we see fit.

The `set_index()` method can be used to do the job. Here is what happens when we `set_index` to the `title` field:
```Python
reviews.set_index("title")
```
This is useful if you can come up with an index for the dataset which is better than the current one.

# Conditional selection
So far we've been indexing various strides of data, using structural properties of the DataFrame itself. To do _interesting_ things with the data, however, we often need to ask questions based on conditions.

For example, suppose that we're interested specifically in better-than-average wines produced in Italy.

We can start by checking if each wine is Italian or not:
```Python
reviews.country == 'Italy'
```
![[Screenshot 2025-01-21 at 11.53.58 PM.png|300]]
This operation produced a Series of `True`/`False` booleans based on the `country` of each record. 

This result can then be used inside of `loc` to select the relevant data:
```Python
reviews.loc[reviews.country == 'Italy']
```

This DataFrame has ~20,000 rows. The original had ~130,000. That means that around 15% of wines originate from Italy.

We also wanted to know which ones are better than average. Wines are reviewed on a 80-to-100 point scale, so this could mean wines that accrued at least 90 points.

We can use the ampersand (`&`) to bring the two questions together:
```Python
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
```

Suppose we'll buy any wine that's made in Italy _or_ which is rated above average. For this we use a pipe (`|`):
```Python
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
```

Pandas comes with a few built-in conditional selectors, two of which we will highlight here.
#### isin()
The first is `isin`. `isin` is lets you select data whose value "is in" a list of values. For example, here's how we can use it to select wines only from Italy or France:
```Python
reviews.loc[reviews.country.isin(['Italy', 'France'])]
```

#### isnull()
The second is `isnull` (and its companion `notnull`). These methods let you highlight values which are (or are not) empty (`NaN`). For example, to filter out wines lacking a price tag in the dataset, here's what we would do:
```Python
reviews.loc[reviews.price.notnull()]
```


### Exercise:
- question #1
Select the first 10 values from the `description` column in `reviews`, assigning the result to variable `first_descriptions`
```Python
first_descriptions = reviews.description.iloc[0:10]
```
Note that many other options will return a valid result, such as `desc.head(10)` and `reviews.loc[:9, "description"]`.

- question #2
Select the records with index labels `1`, `2`, `3`, `5`, and `8`, assigning the result to the variable `sample_reviews`.
```Python
sample_reviews = reviews.iloc[[1,2,3,5,8]]
```

- question #3
Create a variable `df` containing the `country`, `province`, `region_1`, and `region_2` columns of the records with the index labels `0`, `1`, `10`, and `100`. In other words, generate the following DataFrame:
```Python
df = reviews.loc[[0,1,10,100], ["country", "province", "region_1", "region_2"]]

# or
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]
```

Create a variable `df` containing the `country` and `variety` columns of the first 100 records.

Hint: you may use `loc` or `iloc`. When working on the answer this question and the several of the ones that follow, keep the following "gotcha" described in the tutorial:

> `iloc` uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. `loc`, meanwhile, indexes inclusively.

> This is particularly confusing when the DataFrame index is a simple numerical list, e.g. `0,...,1000`. In this case `df.iloc[0:1000]` will return 1000 entries, while `df.loc[0:1000]` return 1001 of them! To get 1000 elements using `loc`, you will need to go one lower and ask for `df.iloc[0:999]`.

```Python
df = reviews.loc[0:99, ["country", "variety"]]
```

- question
Create a DataFrame `top_oceania_wines` containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.
```Python
top_oceania_wines = reviews.loc[(reviews.country.isin(["Australia","New Zealand"])) & (reviews.points>=95)]
```

