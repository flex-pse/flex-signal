# Calibration against a reference

`calibrate` is for a common plant-data situation: one meter (the **target**, A) always reports, but its error
varies, while another (the **reference**, B) measures the same quantity accurately but only some of the time.
Examples are a field flow meter checked against a custody-transfer meter, an online analyser checked against lab
samples, or a cheap sensor next to a precise one that is only switched on now and then.

## The trusted rows

Everything hinges on one idea: the **trusted rows** are the rows where the reference has a value *and* `fit_when`
holds. `fit_when` says when B can be believed, for example `"C == 0 and D == 0 and B > 0"` when a valve C biases B
and B reads 0 while a pump D runs.

The trusted rows are used twice:

1. **To fit.** The model learns to predict B from the `features` (by default, A itself) on the trusted rows.
2. **To prefer the reference.** With `prefer_reference=True` (the default), the trusted rows take B's own value,
   because a good measurement beats a prediction.

Leaving out `fit_when` means "trust B wherever it has a value". If some of those values are wrong, the fit is
wrong everywhere. On the synthetic A/B/C/D data, fitting on every B sample makes A *worse* than not calibrating at
all; restricting the fit to the trusted rows takes the error down to the sensor noise.

## What the output contains

The corrected stream (`output`, by default `<target>_clean`) is a new stream, unless you set `output` to the
target's own name to replace it. Each value is one of:

| Where | Value | Flags |
|---|---|---|
| Trusted rows (with `prefer_reference`) | the reference's reading | the reference's flags |
| `apply_when` holds and the features have values | the model's prediction | the target's flags + `CORRECTED` |
| `apply_when` holds but a feature is missing | missing | the target's flags + `MISSING` |
| `apply_when` is false | the target's value, unchanged | the target's flags |

So an imputer that runs next fills only the rows that are genuinely missing, and a downstream user can always tell
a measured value from a corrected one.

## Choosing a model

The model must be one of the accepted keys. A spec can't name an arbitrary class, because loading a spec must never
import code.

| `model` | Use it when |
|---|---|
| `sklearn:LinearRegression`, `statsmodels:OLS` | the default: a gain and an offset |
| `sklearn:HuberRegressor`, `statsmodels:RLM` | a few trusted rows may still be bad; robust fits down-weight them |
| `sklearn:Ridge` | several correlated features |
| `sklearn:TheilSenRegressor` | heavy outliers; slower |
| `scipy:curve_fit` | the relationship isn't a straight line: give a `formula` and `initial` values |

All of these fit a model that is linear in its coefficients, or a formula, so the fitted state is a few numbers in
JSON. Applying it needs only numpy.

`mode="residual"` predicts the *error*, `B - A`, and adds it to A. Use it when the features explain the error
rather than the value, for example when a temperature biases A:
`features=["temperature"], mode="residual"`.

## Following drift

A static calibration is fitted once, on the training data. If the target drifts, a single gain and offset can't fit
both the start and the end. `refit={"rolling": "1D"}` refits the model for each block of time (`every`, one day by
default) on the trusted rows from the window ending with that block. Blocks whose window has fewer than `min_rows`
trusted rows use the model fitted on the training data.

Rolling refits trade noise for tracking. Each refit sees fewer rows than a static fit, so its coefficients are less
precise. On data that doesn't drift, a rolling refit is a little worse than a static one; on data that does, it is
far better. Pick the window to be as long as possible while the drift within it stays small.

Fitting a gain needs reference data spread over the operating range. A window holding only one short stretch of
reference data, where the flow hardly changes, pins the offset but not the slope. That happens to the first blocks,
whose windows reach back before the data starts, and to windows where gaps removed most of the reference. Make the
window wide enough to hold several reference periods at different operating points. On the synthetic A/B/C/D data,
where B reports for one hour in six, a 2-day window is robust where a 1-day window occasionally isn't.

With a rolling refit, the models are refitted every time the pipeline runs, on the reference data in *that* run.
The model fitted on the training data is the fallback, so a period with no reference data still gets a sensible
correction.

## Checking the fit

`fit()` stores fit metrics next to the coefficients: the number of trusted rows used, the RMSE of the fit, and R².
They appear in the fitted pipeline's state and in `FittedPipeline.report()`, under each step's `fit_metrics`.
