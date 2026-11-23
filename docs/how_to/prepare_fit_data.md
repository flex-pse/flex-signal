# Prepare fit data for a flex-pse regressor

This guide prepares cleaned plant data for fitting a surrogate, fits flex-pse's linear regressor, and scores it on
held-out data. It needs flex-pse (see [Installation](../getting_started/installation.md)).

## 1. Clean the data

```python
import flexsignal as fs
from flexsignal import handoff
from flexsignal.testing import synthetic

case = synthetic.plant(seed=2).with_spikes("flow", count=5).with_spikes("pressure", count=5)
pipe = fs.Pipeline(name="cleanup")
pipe.add_step(fs.clean.Hampel(["flow", "pressure"], window="1h", n_sigma=6))
clean = fs.run(pipe.spec, case.signal()).signal
```

## 2. Prepare `X` and `y`

Pick the inputs and the one output. `fit_data` joins them on time, resamples, and applies a gap rule. The spikes the
cleaning removed are gaps now. Hourly means average over them, and `gaps="drop"` leaves out any hour that is still
empty:

```python
fit = handoff.fit_data(
    clean.select(["flow"]),        # X: one or more inputs
    clean.select(["pressure"]),    # y: one output
    {"resample": {"freq": "1h"}, "gaps": "drop"},
)
print(fit.X.shape, fit.units, fit.steps[-1])
print(fit.data_hash)             # the same data always gives the same hash
```

## 3. Hold out the end, fit, and score

```python
from flexparameterize import LinearRegressor

X_train, y_train, X_test, y_test = handoff.split_holdout(fit.X, fit.y, "1D")
regressor = LinearRegressor().fit(
    X_train, y_train, input_units=fit.input_units, output_units=fit.output_units
)
coefficients = regressor.to_fit_result().coefficients
predicted = coefficients["intercept"] + coefficients["flow"] * X_test["flow"]
print(coefficients, fs.eval.metrics(y_test["pressure"], predicted))
surrogate = regressor.to_surrogate_spec()   # for the flex-pse model
```

## Gap rules

| `gaps` | Effect |
|---|---|
| `error` (default) | Refuse missing values, naming the first |
| `drop` | Leave out every row with a missing value |
| `interpolate` | Fill in time, up to `limit` rows in a row; never past the ends |
| `longest_segment` | Keep only the longest run of complete rows, for regressors that need an unbroken series |

`align="outer"` keeps every timestamp any input has, which leaves gaps for the rule to handle; `"inner"` (the default)
keeps only the timestamps all of them share.

## Window, names and clock

Three more options shape the result for the regressor and the model it feeds:

```python
fit = handoff.fit_data(
    clean.select(["flow"]),
    clean.select(["pressure"]),
    {
        "resample": {"freq": "1h"},
        "gaps": "drop",
        "start": "2025-01-07",                  # keep rows from here (UTC without a zone)
        "end": "2025-01-09",                    # ... up to, not including, here
        "rename": {"flow": "flow_in", "pressure": "pressure_out"},  # the model's names
        "timezone": "America/Los_Angeles",      # naive wall-clock times, as the model runs
    },
)
assert list(fit.X.columns) == ["flow_in"] and fit.X.index.tz is None
```

`timezone` matters for time-series surrogates such as flex-pse's ARIMA, whose fitted history is dated by the index:
it must be on the same naive clock as the model's time steps, as `to_grid` gives them with
`GridOptions(timezone=...)`.
