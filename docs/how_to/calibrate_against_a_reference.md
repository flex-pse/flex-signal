# Calibrate a stream against a reference

This guide corrects a drifting meter A with an accurate meter B that reports only some of the time, and is
untrustworthy while a valve C is open or a pump D runs. See [Calibration](../explanation/calibration.md) for how it
works.

## 1. Look at when the reference can be trusted

```python
import numpy as np
import flexsignal as fs
from flexsignal.core import evaluate_condition
from flexsignal.testing import synthetic

case = synthetic.abcd(seed=1, drift_per_day=2.0)   # A drifts 2 m^3/hr a day
raw = case.signal()

trusted = evaluate_condition(raw, "valid(B) and C == 0 and D == 0 and B > 0")
print(f"B reports on {raw.values['B'].notna().mean():.0%} of rows; "
      f"{trusted.mean():.0%} of rows are trustworthy")
```

Write `fit_when` from what you know about the process. Here B reads high while C is open and reads 0 while D runs.
If B also needs time to settle, wrap the condition in `held(...)`, for example
`held(C == 0 and D == 0, "15min") and B > 0`.

## 2. Calibrate

```python
calibrate = fs.correct.Calibrate(
    target="A",
    reference="B",
    fit_when="C == 0 and D == 0 and B > 0",
    model="sklearn:HuberRegressor",      # robust to the odd bad reference value
)
fitted, out = fs.Pipeline([calibrate]).fit_transform(raw)
print(fitted.states[0]["metrics"])
```

## 3. Check it, and follow drift if you need to

Compare the model's prediction with B on the trusted rows, day by day. (With `prefer_reference` on, those rows
take B's own value, so turn it off for the check.) A trend in the daily mean residual means the target drifts,
and a static calibration can't follow it:

```python
check = calibrate.model_copy(update={"prefer_reference": False})
_, predicted = fs.Pipeline([check]).fit_transform(raw)
residual = (raw.values["B"] - predicted.values["A_clean"])[trusted.to_numpy()]
print(residual.resample("1D").mean().round(1).tolist())   # a steady trend: drift
```

A rolling refit follows the drift. The synthetic data knows the true flow, so we can confirm it:

```python
def rmse(frame):
    error = frame.values["A_clean"] - case.underlying["A"]
    return float(np.sqrt(np.mean(error.to_numpy() ** 2)))

rolling = calibrate.model_copy(update={"refit": fs.correct.RollingRefit(rolling="1D")})
_, followed = fs.Pipeline([rolling]).fit_transform(raw)
print(f"static: {rmse(out):.1f}, rolling: {rmse(followed):.1f} m^3/hr")
assert rmse(followed) < rmse(out)
```

In a spec, the same step is:

```json
{"kind": "calibrate", "target": "A", "reference": "B",
 "fit_when": "C == 0 and D == 0 and B > 0",
 "model": "sklearn:HuberRegressor", "refit": {"rolling": "1D"}}
```

## Variations

- **Replace A instead of adding `A_clean`:** `output="A"`.
- **Correct only some of the time:** `apply_when="D == 1"` corrects A only while D runs, and passes A through
  unchanged elsewhere.
- **The error depends on another stream:** `features=["temperature"], mode="residual"` models `B - A` from the
  temperature and adds it to A.
- **A curve, not a line:** `model="scipy:curve_fit", formula="a + b * A + c * A ** 2",
  initial={"a": 0, "b": 1, "c": 0}`.
