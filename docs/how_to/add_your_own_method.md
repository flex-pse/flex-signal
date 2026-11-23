# Add your own method

This guide registers a cleaner, an imputer, a forecaster and a model of your own, runs them from Python and from a
JSON spec, saves the fitted pipeline, and packages the methods so a `pip install` makes them available. See
[Your own methods](../explanation/extending.md) for the rules behind it.

## 1. A cleaner

A cleaner is a function that takes one stream, as a pandas Series, and returns True for each value to reject. Its
keyword-only parameters, after the `*`, become the step's options:

```python
import numpy as np
import pandas as pd

import flexsignal as fs
from flexsignal.testing import synthetic


def despike(values: pd.Series, *, window: str = "2h", k: float = 8.0) -> pd.Series:
    """Reject values far from a rolling median.

    Args:
        values: The stream.
        window: Width of the rolling window, e.g. 2h.
        k: How many robust standard deviations from the median count as a spike.
    """
    median = values.rolling(window, center=True, min_periods=1).median()
    deviation = (values - median).abs()
    spread = 1.4826 * deviation.rolling(window, center=True, min_periods=1).median()
    return deviation > k * spread.clip(lower=1e-9)


fs.register_cleaner("PlantDespike", despike, flag="OUTLIER")
```

It is now `fs.clean.PlantDespike`, with validated options:

```python
case = synthetic.plant(seed=3).with_spikes("flow", count=8)
result = fs.run(fs.Pipeline([fs.clean.PlantDespike("flow", window="3h")]).spec, case.signal())

caught = result.signal.has_flag(fs.Flag.OUTLIER)["flow"]
spikes = case.fault_mask("spike", "flow")
print(f"caught {int((caught & spikes).sum())} of {int(spikes.sum())} spikes, "
      f"{int((caught & ~spikes).sum())} false alarms")

try:
    fs.clean.PlantDespike("flow", k="large")
except Exception as exc:
    print(type(exc).__name__)
```

## 2. An imputer that learns from training data

An imputer gets every stream (the one being filled first) and a boolean Series of the gaps it may fill, and
returns proposed values; only the gaps are used. An optional `fit` function learns a JSON-able state from the
training data, which the method then receives as `state`:

```python
def fill_from(values: pd.DataFrame, gaps: pd.Series, state: dict, *, source: str) -> np.ndarray:
    """Fill a stream from another one, scaled by their training ratio.

    Args:
        values: The streams, the one being filled first.
        gaps: Where to fill.
        state: The fitted ratio.
        source: The stream to fill from.
    """
    return state["ratio"] * values[source].to_numpy()


def fit_fill_from(values: pd.DataFrame, *, source: str) -> dict:
    target = values.iloc[:, 0]
    both = target.notna() & values[source].notna()
    return {"ratio": float(target[both].mean() / values[source][both].mean())}


fs.register_imputer("FillFromStream", fill_from, fit=fit_fill_from)
```

flex-signal finds the gaps, respects `max_gap`, and flags what it fills `IMPUTED_MODEL`. The imputer works as a
`tiered` tier too, by its name:

```python
tiered = fs.impute.Tiered(
    "flow",
    tiers=[
        {"max_gap": "15min", "method": "interpolate"},
        {"max_gap": None, "method": "FillFromStream", "options": {"source": "reference_flow"}},
    ],
)
```

## 3. A forecaster and a model

A forecaster gets the target's history (with any `exog` streams) up to the issue time, and the `exog` streams
over the forecast rows. It returns the mean, and a `std` or quantiles if you want intervals:

```python
def same_hour_mean(history: pd.DataFrame, future: pd.DataFrame, *, days: int = 7) -> dict:
    """The mean of the same hour over the last few days.

    Args:
        history: The target, up to the issue time.
        future: The forecast's rows.
        days: How many days back to average.
    """
    y = history.iloc[:, 0]
    recent = y[y.index > y.index[-1] - pd.Timedelta(days=days)]
    by_hour = recent.groupby(recent.index.hour)
    hours = future.index.hour
    return {"mean": by_hour.mean().reindex(hours).to_numpy(),
            "std": by_hour.std().reindex(hours).to_numpy()}


fs.register_forecaster("SameHourMean", same_hour_mean)
```

A model is a regressor class that works like scikit-learn's and is described by its fitted `coef_` and
`intercept_`. Registered, it is a `model` choice for `calibrate` and `regression`:

```python
from sklearn.linear_model import Ridge


class StrongRidge(Ridge):
    def __init__(self, alpha: float = 10.0):
        super().__init__(alpha=alpha)


fs.register_model("StrongRidge", StrongRidge, description="Ridge with alpha 10.")
```

## 4. Use them from a JSON spec

Once registered, the methods are used by name. The spec names nothing to import; `requires` only tells someone
without the methods what to install:

```python
spec = {
    "schema_version": "0.1.0",
    "name": "my_methods",
    "requires": ["myplant.methods"],
    "steps": [
        {"kind": "PlantDespike", "streams": ["flow"], "window": "3h"},
        {"kind": "FillFromStream", "streams": ["flow"], "source": "pressure", "max_gap": "6h"},
        {"kind": "regression", "streams": ["pressure"], "features": ["temperature"],
         "model": "StrongRidge"},
        {"kind": "resample", "freq": "1h"},
        {"kind": "SameHourMean", "streams": ["flow"], "horizon": "1D", "quantiles": [0.1, 0.9]},
    ],
}
data = synthetic.plant(seed=3, periods=10 * 288).with_spikes("flow", count=8).with_gaps("flow")
result = fs.run(spec, data.signal())
print(result.provenance[0]["function"])
```

Each history record names the function behind the step. A spec with an option of the wrong type fails when it
loads, naming the field:

```python
bad = {**spec, "steps": [{"kind": "PlantDespike", "streams": ["flow"], "k": "large"}]}
try:
    fs.load_spec(bad)
except fs.SignalConfigError as exc:
    print(exc.field)  # steps.0.k
```

## 5. Save and reload the fitted pipeline

The fitted state is JSON, so the pipeline saves like any other. Loading it needs the same methods registered
first:

```python
import tempfile
from pathlib import Path

folder = Path(tempfile.mkdtemp()) / "my_methods"
result.fitted.save(folder)
reloaded = fs.FittedPipeline.load(folder)
again = reloaded.transform(data.signal())
assert again.values.equals(result.signal.values)
```

## 6. Package the methods

Put the functions and the `fs.register_*` calls in a module of a package, for example `myplant/methods.py`, and
expose the module through the `flexsignal.steps` entry-point group in the package's `pyproject.toml`:

```{code-block} toml
[project.entry-points."flexsignal.steps"]
myplant = "myplant.methods"
```

Once the package is installed, its methods are available wherever flex-signal runs, including Ray workers.
Without an entry point, load the module from your code:

```{code-block} python
fs.load_plugins("myplant.methods")
result = fs.run(spec, data, plugins=["myplant.methods"])  # the same, per call
```
