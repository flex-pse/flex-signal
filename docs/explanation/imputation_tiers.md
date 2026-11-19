# Imputation tiers

An imputer fills missing values. Which one to use depends mostly on **how long the gap is**. A 10-minute dropout
is best bridged with a line; a 6-hour outage needs a model that knows the daily cycle or a related stream.

## What every imputer has in common

- **A gap is a run of missing values.** Its length is the time between the valid samples either side of it. On
  5-minute data, two missing samples make a 15-minute gap. A gap at the start or end of the data has no length.
- **`max_gap`** is the longest gap an imputer fills. Longer gaps are left alone, for a later step or for good.
  With `max_gap` unset, a model imputer fills every gap, including those at the start and end.
- **Deliberate exclusions are never filled.** Values a `mask` step flagged `EXCLUDED`, and rows that a join of
  several inputs added (`NO_SAMPLE`), stay missing.
- **Filled values say so.** They keep the flags that explain why they were missing, and gain `IMPUTED_SHORT`
  (interpolation) or `IMPUTED_MODEL` (any model).

## The methods

| Method | How it fills | Good for | Learns at `fit` |
|---|---|---|---|
| `interpolate` | a line or smooth curve between the values either side | short gaps | nothing |
| `kalman` | the Kalman smoother of a level (or trend) plus daily or weekly cycles | gaps of any length in one stream with a regular pattern | the model's variances |
| `regression` | a linear model of other streams, their lags, and cycles | long gaps, when a related stream is still reporting | coefficients |
| `knn` | the average of the most similar rows | several related streams | nothing (uses the data it fills) |
| `iterative` | each stream modelled from the others, round-robin | several related streams | nothing (uses the data it fills) |

A few notes:

- `kalman` needs a regular time index, as do `regression` lags and the scipy filters. Run `align` or `resample`
  first. Give it `seasonal` cycles when the stream has a daily pattern. Without them, a long gap relaxes towards a
  flat line.
- `regression` can't use the stream it fills as a feature, because that stream is missing exactly where the model
  is needed. Use `kalman` to fill a stream from its own history.
- `knn` and `iterative` learn from the data they fill, every time they run. They are convenient but not fitted
  models: their fill for a row can change when other rows change.

Deep-learning imputers (SAITS, BRITS and others, through PyPOTS) are in the `[pypots]` extra:
`fs.impute.PypotsImpute`, which can also be a tier.

## Tiers

`tiered` gives each gap to the method that suits its length:

```python
import flexsignal as fs

step = fs.impute.Tiered(
    "flow",
    tiers=[
        {"max_gap": "15min", "method": "interpolate"},
        {"max_gap": "6h", "method": "regression", "options": {"features": ["pressure"]}},
        {"max_gap": None, "method": "kalman",
         "options": {"seasonal": [{"period": "1D", "harmonics": 2}]}},
    ],
)
```

A gap goes to the first tier whose `max_gap` covers it. Gap lengths are measured once, in the data as it arrives at
the step, so a long gap never becomes "short" because a tier filled part of it. Every tier's imputer is fitted on the
training data, and its state is kept under `tiers` in the fitted pipeline.

Where should the boundaries go? Measure them: the [imputer benchmark](../how_to/evaluate_imputers.md) hides known
values in your data and scores each method by gap length.
