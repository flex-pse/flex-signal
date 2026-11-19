# flex-signal in ten minutes

This page solves a typical plant-data problem end to end. Every code block runs as written.

**The problem.** Meter **A** always reports, but it is miscalibrated: it reads 15% low with an offset. Meter **B**
measures the same flow accurately, but it reports only one hour in every six. B also can't always be trusted:
it reads high while valve **C** is open and reads 0 while pump **D** runs. We want a clean, complete,
correctly calibrated A.

## 1. The data

flex-signal ships a synthetic version of this problem, with the true flow known, so we can check the answer:

```python
import numpy as np
import flexsignal as fs
from flexsignal.testing import synthetic

case = synthetic.abcd(seed=0)       # 14 days of 5-minute data: streams A, B, C, D
case = case.with_gaps("A", count=6, length=24)   # and some 2-hour dropouts in A
raw = case.signal()                 # with your own data: fs.SignalFrame.from_frame(df, units=...)
true_a = case.underlying["A"]

def error(values):
    """RMSE against the true flow, over the rows that have a value."""
    return float(np.sqrt(np.nanmean((values - true_a).to_numpy() ** 2)))

print(f"raw A is off by {error(raw.values['A']):.1f} m^3/hr")
```

## 2. The pipeline

Three steps, each with one job:

```python
pipe = fs.Pipeline(
    [
        # B is a flow: a negative reading is an error code, not data.
        fs.clean.Range("B", min=0),
        # Learn A -> B where B is trustworthy, then correct A everywhere.
        fs.correct.Calibrate(
            target="A",
            reference="B",
            fit_when="C == 0 and D == 0 and B > 0",
            model="sklearn:HuberRegressor",
            output="A_clean",
        ),
        # Fill the gaps: short ones by interpolation, long ones with a model.
        fs.impute.Tiered(
            "A_clean",
            tiers=[
                {"max_gap": "15min", "method": "interpolate"},
                {"max_gap": None, "method": "kalman",
                 "options": {"seasonal": [{"period": "1D", "harmonics": 2}]}},
            ],
        ),
    ],
    name="a_cleanup",
)
```

`fit_when` is a [condition](../explanation/conditions.md). It decides which rows teach the calibration. Without it,
the rows where B reads high or reads 0 would drag the fit off.

## 3. Fit and run

```python
fitted, clean = pipe.fit_transform(raw)

print(f"A_clean is off by {error(clean.values['A_clean']):.2f} m^3/hr")
print(clean.flag_counts()["A_clean"]["flags"])
print(fitted.states[1]["metrics"])          # how well the calibration fitted
assert error(clean.values["A_clean"]) < 1.5
assert clean.values["A_clean"].notna().all()
```

Every value of `A_clean` says where it came from:

- no flags: B's own reading, on the rows where B is trusted (`prefer_reference`, on by default);
- `CORRECTED`: A, calibrated;
- `CORRECTED`, `MISSING` and `IMPUTED_*`: A was missing there, and the gap was filled.

## 4. Use it again

The fitted pipeline is plain JSON: the spec plus the calibration coefficients and the Kalman model's variances.
Apply it to next week's data without refitting:

```python
fitted.save("a_cleanup", overwrite=True)
again = fs.FittedPipeline.load("a_cleanup").transform(raw)
assert again.equals(clean)
```

## Next

- The [tutorials](../tutorials/index.md) build this up step by step. Tutorial 05 is this problem, with drift added.
- [Calibration](../explanation/calibration.md) explains the trusted rows, the models, and rolling refits.
- [Imputation tiers](../explanation/imputation_tiers.md) explains which imputer suits which gap.
