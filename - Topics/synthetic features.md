A [**feature**](https://developers.google.com/machine-learning/glossary#feature) not present among the input features, but assembled from one or more of them. Methods for creating synthetic features include the following:
- [**Bucketing**](https://developers.google.com/machine-learning/glossary#bucketing) a continuous feature into range bins.
- Creating a [**feature cross**](https://developers.google.com/machine-learning/glossary#feature_cross).
- Multiplying (or dividing) one feature value by other feature value(s) or by itself. For example, if `a` and `b` are input features, then the following are examples of synthetic features:
    - ab
    - a$^2$
- Applying a transcendental function to a feature value. For example, if `c` is an input feature, then the following are examples of synthetic features:
    - sin(c)
    - ln(c)

Features created by [**normalizing**](https://developers.google.com/machine-learning/glossary#normalization) or [**scaling**](https://developers.google.com/machine-learning/glossary#scaling) alone are not considered synthetic features.