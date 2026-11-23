# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). What stays stable between releases, and how features are deprecated,
is described in [Stability and deprecation](https://flex-pse.github.io/flex-signal/explanation/stability.html).

## [Unreleased]

## [0.1.0] - unreleased

The first release. Pipelines select, clean, calibrate, impute and forecast plant timeseries data, from Python or
from JSON specs, and hand the result to flex-pse models and flex-orchestrator studies.

### Spec and stability

- **`SignalSpec`, schema `0.1.0`, frozen.** Strict pydantic models with `kind`-discriminated steps, named input
  slots (`frame`, `parquet`, `csv`) and outputs (`parquet`, `csv`). `load_spec` / `dump_spec`, `$source` file
  references, and `spec_hash` over canonical JSON (the same whatever the key order or written defaults).
  - A step *is* its spec: the object built in Python is what JSON loads into, so both give bit-identical output.
    Fitted state is a separate JSON dict.
  - Errors name the exact field, e.g. `steps.2.window: '1 hour' is not a duration ...`, and carry it in
    `error.field`.
  - The JSON Schema is committed at `src/flexsignal/spec/schemas/signal_spec.schema.json`, and
    `flexsignal.spec.build_json_schema()` returns it.
  - Specs and fitted pipelines saved by this release will keep loading in later ones: later schema versions
    upgrade them through `MIGRATIONS`. The test suite holds a frozen copy of the 0.1.0 schema, a spec with every
    step kind and a fitted payload, and fails if the schema changes without a new version and a migration.
  - A spec validated directly, such as one embedded in a caller's pydantic model, has its version checked and is
    migrated as `load_spec` does.
- **Conditions can't run code.** `keep_when`, `fit_when`, `invalid_when` and the other conditions are parsed and
  checked against an allowlist (stream names, numbers, arithmetic, comparisons, boolean operators and the
  functions in `CONDITION_FUNCTIONS`) when a spec loads, then evaluated by flex-signal with numpy. Helpers:
  `held(C == 0, "30min")`, `valid(B)`, `is_raw(B)`, `flagged(A, "OUTLIER")`. Every comparison with a missing value
  is False, `!=` included.
- **Specs never import code.** Models and methods are named by registered keys, never `"package.module:Class"`
  paths.
- A deprecation policy: deprecated features warn for at least one minor release before they are removed.

### Data model

- `SignalFrame`: values, a per-sample `uint16` quality-flag bitmask, per-stream `StreamMeta` (units, quantity
  kind, source, forecast issue time) and provenance `history`. The index is tz-aware UTC, and frames are
  immutable.
- `Flag` bits: `MISSING`, `OUT_OF_RANGE`, `RATE_OF_CHANGE`, `OUTLIER`, `STUCK`, `MANUAL`, `EXCLUDED`, `CORRECTED`,
  `IMPUTED_SHORT`, `IMPUTED_MODEL`, `FORECAST`, `DERIVED`, `RESAMPLED`, `NO_SAMPLE`, `FILTERED`. Cleaning sets a
  value to NaN and a flag; it never deletes data. Rows a stream gets only from joining inputs are `NO_SAMPLE`.
- Timestamps without a time zone are taken as the machine's local time, with a `NaiveTimestampWarning`;
  `timezone=` sets it. Timestamps written in a spec mean UTC.
- The `SignalError` hierarchy (`SignalConfigError`, `SignalDataError`, `SignalBackendError`), with dotted `field`
  paths, surviving pickling.
- Parquet I/O that keeps flags, units and provenance in the Arrow metadata, and CSV with exact float round trips.
- `[units]`: `convert_units` converts streams with pint, and a `range` bound may be given in other units.

### Steps

- **select:** `align`, `resample`, `window`, `mask`, `rename`, `combine`, `shift`, `convert_units`, and `scale`
  (multiply by a constant and set new units, e.g. through a density). `combine(how="sum")` is missing wherever an
  input is missing, unless `missing_as_zero=True`.
- **clean:** `range`, `rate_of_change` (with `reset_after`, so a bad first value or a genuine fast step doesn't
  reject everything after it), `stuck`, `hampel`, `tukey` (fences learned in `fit`), `expression`,
  `filter` (Savitzky-Golay, Butterworth, moving median), `detotalize` (running total to rate, with rollovers).
- **correct:** `calibrate` fits a stream to a trusted, intermittent reference where `fit_when` holds and applies
  it where `apply_when` holds, prefers the reference where it is trusted, and can refit on a rolling window to
  follow drift. Models: scikit-learn's `LinearRegression`, `Ridge`, `HuberRegressor`, `TheilSenRegressor`,
  statsmodels' `OLS` and `RLM`, and `scipy:curve_fit` with a formula. Fitted models are JSON coefficients.
- **impute:** `interpolate` (linear, nearest, previous, PCHIP, Akima), `kalman`, `regression` (other streams,
  lags, Fourier terms), `knn`, `iterative`, and `tiered`, which routes each gap to a method by its length. Every
  imputer takes `max_gap` and never fills `EXCLUDED` or `NO_SAMPLE` values. Daily or weekly cycles are one
  `CycleTerm` (`period`, `harmonics`) wherever a step takes them.
- **forecast:** `persistence` and `seasonal_naive` baselines, `structural` (statsmodels' `UnobservedComponents`)
  and `regression`. A forecast writes the frame's future rows, flagged `FORECAST`, from an `issue_time`; `exog`
  streams are known ahead, and `quantiles` add `<stream>__q10`-style streams.

### Your own methods and adapters

- `fs.register_cleaner`, `fs.register_imputer` and `fs.register_forecaster` turn a function into a step that runs
  from Python and JSON, in `tiered` and in backtests, and saves with a fitted pipeline. Its keyword-only
  parameters become validated options. `fs.register_model` adds a regressor for `calibrate` and `regression`.
  Only registered methods run; `fs.load_plugins`, `plugins=` and the `flexsignal.steps` entry-point group
  register them.
- Adapters, registered the same way and imported only when a step runs:
  - `[anomaly]`: `PyodDetector`, `RupturesChangepoint`, `StumpyDiscord`;
  - `[pypots]`: `PypotsImpute` and `PypotsForecast`, with windowing that restores the index;
  - `[foundation]`: `ChronosForecast` (Chronos-2, and TimesFM 2.5 if installed); `[t0]`: `T0Forecast` (t0-alpha).
    Pinned revisions, open (Apache-2.0) weights only. `[t0]` can't be installed alongside `[pypots]`, so `[all]`
    leaves it out;
  - `[deep]`: `NeuralforecastForecast`; `[darts]`: `DartsForecast`; `[sktime]`: `SktimeForecast`.
  - An adapter's `model` takes any model class of its library by name; untested ones take their settings from
    `params`. Learned models train on the data they are given and store nothing.

### Pipelines, evaluation and hand-off

- `Pipeline` (with `add_step`) and `FittedPipeline`: `fit`, `transform`, `report`, `to_payload` /
  `from_payload`, and `save` / `load` as plain JSON files, no pickle. Each step's effect is recorded in the
  provenance history.
- `fs.run`, `fs.fit` and `fs.backtest` keep the orchestrator contract: spec in, JSON-able results out, no I/O when
  data is passed, `(df, units)` input slots, progress callbacks, picklable results and errors. Tested in
  `src/flexsignal/tests/contract/`.
- `fs.backtest`: blocked, time-ordered folds that hide everything except `exog` after each issue time, with other
  forecasters swapped in for comparison and skill against a baseline.
- `flexsignal.eval`: `benchmark_imputers` (hidden gaps of given lengths, scored by RMSE, MAE, bias and coverage),
  and `metrics` (RMSE, MAE, MAPE).
- `flexsignal.handoff`: `to_grid` (one value per model time step, with a facility time zone, interval rules and
  gap rules; `GridValues.resampled` counts steps resting on resampled or upsampled samples, and
  `to_overrides(strict=True)` requires a model path for every column), `data_window`, `fit_data` (with a
  `start`/`end` window, a `rename` to the regressor's names, and a `timezone` for the model's wall clock) and
  `split_holdout` for surrogate fits, and `grids_of` / `scope_of` /
  `targets` for models with several time blocks. They carry flex-orchestrator's own behaviour and tests.
- `flexsignal.integrations.flexpse`: `time_grid`, `time_scopes` and `apply_values` for flex-pse specs and built
  models, against the latest flex-pse release (0.3.1). `import flexsignal` never imports flex-pse, pyomo, idaes or
  torch.
- `flexsignal.testing.synthetic`: seeded plant-like data, the A/B/C/D calibration problem, and an anaerobic digester
  (biogas from a fast and a slowly decaying pool of feed, an offset gas meter and a few lab checks a day), with
  injected faults and exact ground truth.

### Docs and tutorials

- Getting started, explanation pages (including the orchestrator contract and stability), how-to guides
  (including upgrading a spec), an included-methods page and a full API reference with tables generated from the
  code. Every code block on these pages runs in CI.
- Eleven marimo tutorials, from a first look at the data to running a spec as a flex-orchestrator study. Each
  asserts what it claims; CI runs them, and the site publishes them as static pages.

### Not included

- skforecast, statsforecast and mlforecast adapters: they require `pandas<3`, and flex-signal and flex-pse require
  pandas 3.
- Acquirium sources, sinks and apps.
