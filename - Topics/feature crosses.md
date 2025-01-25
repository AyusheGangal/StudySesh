A [**synthetic feature**](https://developers.google.com/machine-learning/glossary#synthetic_feature) formed by "crossing" [**categorical**](https://developers.google.com/machine-learning/glossary#categorical_data) or [**bucketed**](https://developers.google.com/machine-learning/glossary#bucketing) features.

For example, consider a "mood forecasting" model that represents temperature in one of the following four buckets:
- `freezing`
- `chilly`
- `temperate`
- `warm`

And represents wind speed in one of the following three buckets:
- `still`
- `light`
- `windy`

Without feature crosses, the linear model trains independently on each of the preceding seven various buckets. So, the model trains on, for example, `freezing` independently of the training on, for example, `windy`.

Alternatively, you could create a feature cross of temperature and wind speed. This synthetic feature would have the following 12 possible values:
- `freezing-still`
- `freezing-light`
- `freezing-windy`
- `chilly-still`
- `chilly-light`
- `chilly-windy`
- `temperate-still`
- `temperate-light`
- `temperate-windy`
- `warm-still`
- `warm-light`
- `warm-windy`

Thanks to feature crosses, the model can learn mood differences between a `freezing-windy` day and a `freezing-still` day.

If you create a synthetic feature from two features that each have a lot of different buckets, the resulting feature cross will have a huge number of possible combinations. For example, if one feature has 1,000 buckets and the other feature has 2,000 buckets, the resulting feature cross has 2,000,000 buckets.

Formally, a cross is a [Cartesian product](https://wikipedia.org/wiki/Cartesian_product).

Feature crosses are mostly used with linear models and are rarely used with neural networks.