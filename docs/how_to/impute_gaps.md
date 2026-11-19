# Impute gaps

This guide fills a flow meter's gaps: short dropouts by interpolation, and outages of several hours from a related
pressure reading. See [Imputation tiers](../explanation/imputation_tiers.md) for what each method does.

## 1. Clean first, then fill

An imputer fills NaNs, so run it after the cleaners have turned bad values into NaNs. Here the synthetic plant has
short and long gaps in `flow`:

```python
import flexsignal as fs
from flexsignal.testing import synthetic

case = (
    synthetic.plant(seed=5)
    .with_gaps("flow", count=5, length=2)      # 15-minute gaps
    .with_gaps("flow", count=3, length=48)     # 4-hour outages
)
raw = case.signal()
```

## 2. Choose a method per gap length

```python
fill = fs.impute.Tiered(
    "flow",
    tiers=[
        {"max_gap": "30min", "method": "interpolate", "options": {"method": "pchip"}},
        {"max_gap": "12h", "method": "regression",
         "options": {"features": ["pressure"], "fourier": [{"period": "1D"}]}},
    ],
)
fitted, out = fs.Pipeline([fill]).fit_transform(raw)
counts = out.flag_counts()["flow"]["flags"]
print(counts)
assert counts["IMPUTED_SHORT"] == 5 * 2 and counts["IMPUTED_MODEL"] == 3 * 48
```

Gaps longer than the last tier's `max_gap` stay missing. Set the last tier's `max_gap` to `None` to fill every
gap that's left.

## 3. Keep what must not be filled

Data you excluded on purpose stays excluded. If the pump is off, a `mask` step sets the flow aside, and no imputer
fills it:

```python
pipe = fs.Pipeline([fs.select.Mask(keep_when="pump_on == 1", streams=["flow"]), fill])
```

## 4. Single methods

Each tier method is also a step of its own:

```python
fs.impute.Interpolate("flow", max_gap="15min")
fs.impute.Kalman("flow", seasonal=[{"period": "1D", "harmonics": 2}], max_gap="1D")
fs.impute.Regression("flow", features=["pressure"], lags=["5min"])
fs.impute.KNN("flow", features=["pressure", "temperature"])
fs.impute.Iterative("flow", features=["pressure", "temperature"])
```

To pick the methods and boundaries for your own data, [evaluate the imputers](evaluate_imputers.md) on it.
