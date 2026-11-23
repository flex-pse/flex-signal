# Handing data to flex-pse models

flex-signal cleans, corrects and fills timeseries. A flex-pse model then needs that data in two shapes:

- **values per model time step**, to fix a variable (a feed flow, a demand) or set a price;
- **a prepared `X, y`**, to fit a surrogate (a pump's power from its flow).

`flexsignal.handoff` produces both, and `flexsignal.integrations.flexpse` puts values into a model. Everything works
in a plain script; flex-orchestrator calls the same functions for its studies.

## The model's clock

A flex-pse model runs on the **facility's wall clock**. Its time points are naive local times, each the *start* of a
step, and it charges a rate as `rate[t] × dt`. flex-signal keeps data in UTC. So
{func}`~flexsignal.handoff.to_grid` converts:

1. UTC → the facility's `timezone`, then drops the zone. The default is `"UTC"`; set it to the plant's zone, such as
   `"America/Los_Angeles"`. It is a deliberate choice that must not depend on the machine running the code, so it is
   never guessed.
2. Shift by `offset` (`"-365D"`: last year's data for this horizon) or by `data_years` (`{2030: 2025}`: the same
   calendar dates in 2025).
3. Resample onto the model's steps. A step `[t, t+dt)` takes the samples inside it (`window="forward"`, the
   default) and combines them with `agg` (mean by default). `window="backward"` takes `(t-dt, t]`, for values that sum
   up what happened *up to* `t`. With `agg="sum"`, data coarser than the steps, such as hourly kWh on 15-minute steps,
   is split evenly over the steps it covers.
4. Fill steps left empty with the `gaps` rule: `interpolate` (in time, never past the ends), `ffill`, `zero`, or
   `error`. `limit` caps how many steps in a row are filled.

Or, with `reduce`, the horizon's data becomes one number (its mean, last value, maximum, ...).

On days when daylight saving starts, the skipped local hour has no samples and is filled by the gap rule; when it
ends, the repeated hour's samples average into one step.

Only data near the horizon is used: the horizon plus a day each side, so end gaps can be interpolated from the
samples beyond them. {func}`~flexsignal.handoff.data_window` says which UTC window that is, so you can read just that
much from a database.

## Clean first, then hand off

`to_grid`'s gap rules are a last resort for the odd empty step. Clean and fill the data with flex-signal first, where
each method suits the gap ([Imputation tiers](imputation_tiers.md)) and each value is flagged. Then hand off with
`gaps="error"`, so nothing is filled silently. Given a `SignalFrame`, `to_grid` also reports
{attr}`~flexsignal.handoff.GridValues.estimated`: how many steps rest on samples flex-signal filled, corrected or forecast,
and {attr}`~flexsignal.handoff.GridValues.resampled`: how many rest on samples put on a new time grid.

## Into the model

{meth}`GridValues.to_overrides <flexsignal.handoff.GridValues.to_overrides>` names each column by its model path, with
its units (`{"value": [...], "units": "m^3/hr"}`). {func}`~flexsignal.integrations.flexpse.apply_values` takes
them on a spec or on a built model, and the model ends up the same either way:

- On a **spec**, it returns a copy with one `fix` element per value, to build. A path into the `costing` element,
  such as `costing.energy_prices.electrical`, sets that field instead: that's how a flat price is set.
- On a **built model**, it fixes the variables in place through flex-pse's `apply_spec`, checking them against the
  variables' bounds. A path naming a mutable Param, such as a price the model was built with as a Param, sets it.
  flex-pse builds a spec's flat price as a constant, so changing that one needs the spec and a rebuild; the error
  says so.

{func}`~flexsignal.integrations.flexpse.time_grid` gives the model's time points, from a flowsheet spec or a built
model.

## Models with several time blocks

A model can hold several time blocks, for example one per plant. Each block's holder is a **scope**, and a model
path belongs to the nearest time block above it. {func}`~flexsignal.handoff.targets` says where a path is set (a path
under no scope is set in every one), `time_grid` gives one grid per scope, and `apply_values` fixes each group of
paths against its own block. Line data up with each scope's grid separately. Design studies, with one horizon per
year and month, are flex-orchestrator's job; it calls these functions once per horizon.

## Fit data

{func}`~flexsignal.handoff.fit_data` joins input and output frames (`align="inner"` keeps the timestamps they share),
optionally resamples them, and applies a gap rule (`error`, `drop`, `interpolate`, or keep the `longest_segment`
without gaps). It logs every step and hashes the result, so a fit can say exactly which data it used.
{func}`~flexsignal.handoff.split_holdout` keeps the end of the data for testing, and
{func}`flexsignal.eval.metrics` scores predictions.

flex-signal stops there: you call flex-pse's regressor yourself, with
{attr}`~flexsignal.handoff.FitData.input_units` and {attr}`~flexsignal.handoff.FitData.output_units`.

## flex-pse stays optional

`import flexsignal` and `flexsignal.handoff` never import flex-pse, pyomo or idaes. Only
`flexsignal.integrations.flexpse` uses them, inside its functions. flex-pse isn't on PyPI yet, so it is installed
from its git tag (see [Installation](../getting_started/installation.md)).
