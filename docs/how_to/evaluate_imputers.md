# Evaluate imputers

Real gaps have no ground truth, so you can't score an imputer on them directly. The benchmark makes gaps where the
values *are* known: it hides runs of valid samples, lets each imputer fill them, and compares the fill with what was
there.

## 1. Start from clean data on a regular grid

Hide values only where the data is good, so the score measures the imputer and not the faults. Run your cleaning
pipeline first, and `align` or `resample` if the data isn't on a grid.

```python
import flexsignal as fs
from flexsignal.eval import benchmark_imputers, summarize
from flexsignal.testing import synthetic

clean = synthetic.plant(seed=2).signal()
```

## 2. Choose the candidates and the gap lengths

Give each candidate a name. Use the same streams and options you would use in the pipeline. Set `max_gap` wide, so
every method attempts every gap:

```python
candidates = {
    "linear": fs.impute.Interpolate("flow", max_gap="1D"),
    "kalman": fs.impute.Kalman("flow", seasonal=[{"period": "1D", "harmonics": 2}]),
    "regression": fs.impute.Regression("flow", features=["pressure"]),
}
results = benchmark_imputers(
    clean, candidates, streams=["flow"], gaps=["15min", "1h", "6h"], count=5
)
```

Gap lengths are measured the way imputers measure them, between the valid samples either side, so a `"15min"`
gap on 5-minute data hides two samples. `count` gaps are hidden in each stream for each length.

## 3. Read the results

```python
print(summarize(results).round(2))            # RMSE: one row per imputer, one column per gap
print(summarize(results, "coverage"))         # the fraction of hidden values each one filled
```

`results` has one row per imputer, stream, gap length and repeat, with the RMSE, MAE and bias of the filled
values. Look for where the ranking changes: that is where a [tiered](impute_gaps.md) imputer should switch
methods. Here interpolation is close to the best on short gaps but falls far behind on long ones, where the
Kalman smoother stays accurate.

## Tips

- Use `repeats=3` or more for a steadier estimate; each repeat hides the gaps in new places.
- `pattern="random"` hides single samples at random, never two side by side, instead of runs. It mimics scattered dropouts, but it says
  little about outages.
- An imputer that leaves a gap unfilled has `coverage` below 1, and its errors cover only what it filled. Compare
  methods at full coverage.
