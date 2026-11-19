# flex-signal: plan and architecture

Status: draft v0.1 (2026-11-06). Nothing is implemented yet; this document sets direction and dependencies.

flex-signal is the timeseries layer of the flex-pse ecosystem. It takes raw plant data (from files or from
Acquirium), then **selects, cleans, corrects, imputes and forecasts** it. It emits the result in forms that
flex-pse models, flexparameterize, and flex-orchestrator studies can use directly.

It is a **wrapper**. The algorithms come from pandas, scipy, scikit-learn, statsmodels, PyPOTS and the other
libraries listed in §6. flex-signal supplies:

1. a declarative, serializable spec for multi-stream cleaning and forecasting pipelines;
2. a pandas-native step protocol with thin adapters over existing libraries;
3. per-sample quality flags and provenance, so every changed value can be explained;
4. adapters to flex-pse, Acquirium and flex-orchestrator.

---

## 1. Goals and non-goals

**Goals**

- G1: Build complex conditional cleaning and correction logic from simple parts. For example: "correct A using B,
  fit only where C == 0, D == 0 and B > 0".
- G2: Offer tiered imputation, from simple interpolation through statistical models to deep models (PyPOTS),
  behind one interface, with masking-based evaluation.
- G3: Wrap statistical, ML and foundation-model forecasters behind one interface, with backtesting.
- G4: Follow the flex-pse spec conventions: pydantic v2, `schema_version`, `kind` discriminators, JSON on disk,
  `$source` refs, migrations, and a committed JSON Schema. A pipeline can then be emitted, versioned, stored and
  re-run by flex-orchestrator.
- G5: Provide one-call adapters:
  - cleaned or forecast data → values on a flex-pse model's time grid, as `fix` elements (M3, M4; D24),
    and ARIMA history once flex-pse releases it;
  - cleaned data → a prepared `X, y` for flex-pse's regressors (M3);
  - Acquirium streams in (raw data) and out (cleaned data, forecasts, soft-sensor Apps).

**Non-goals**

- Reimplementing algorithms that already exist (interpolation, Kalman smoothing, ARIMA, deep imputers, PyOD
  detectors). The only exception is checks too small to justify a dependency: Hampel filter, range checks,
  rate-of-change checks and stuck-sensor detection, each about 15 lines.
- Running a data store or historian. Acquirium and Parquet files cover storage.
- Owning optimization or scheduling, which belong to flex-pse and flex-orchestrator.

---

## 2. What the ecosystem already gives us (research summary)

| Where | What exists | Consequence for flex-signal |
|---|---|---|
| flex-pse `flexcore.config` | `_StrictModel` (`extra="forbid"`), `FlowsheetSpec` with `schema_version` and a `kind`-discriminated element union, `load_spec`, `dump_spec`, `MIGRATIONS`, `export_json_schemas`, `SourceRef {"$source": ...}`, `Quantity {value, units}` | Copy this pattern exactly for `SignalSpec`. |
| flex-pse `flexcore.exceptions` | `FlexError` → `FlexConfigError(field=, value=)` and `FlexDataError`. Every message says what was wrong and what to do | Mirror the hierarchy and message rule in `flexsignal.core.exceptions` without importing flexcore (see §4.1 on import direction). |
| flex-pse `flexops` | `apply_spec(model, elements)` applies `fix`, `unfix` and `dispatch` to a live model. `set_external_dispatch(var, series)`. Price and exogenous profiles are construction options, set before build. `TimeBlock` uses a naive local-time index, so `index_of` raises on off-grid timestamps | `integrations.flexpse` builds `fix` elements on `TimeBlock.datetime_index` (`apply_values`, on a spec before build or a model after), taking over flex-orchestrator's versions (D22). |
| flex-pse `flexparameterize` (0.3.1, the latest release) | `TagMap`, `check_sufficiency`, `apply_to_model(model, df, tagmap)`, a `Regressor` protocol, and the `constant_intensity` and `multilinear` regressors. ARIMA is still reserved in 0.3.1. On flex-pse's `arima_integration` branch, `ArimaRegressor` **refuses NaNs and uneven steps** and tells the user to "resample and impute" | flex-signal is that missing "resample and impute" step. Our output must be a regular-grid DataFrame with no gaps and columns named by model alias. |
| flex-pse `flexparameterize/tests/test_time_series_data/` (on the `arima_integration` branch; not in any release yet) | Public hourly biogas data (`bio_gas_generation.csv`: feed volume, TS %, biogas m³/h; ~1,440 rows), its 15-minute linear interpolation, and the `impute_timeseries.py` script that made it | The **one permitted outside dataset** (see §8, "Test data"). flex-signal's `interpolate` step supersedes the script, and must reproduce its output. |
| flex-orchestrator (pins flex-pse 0.3.1) | Plugin system: entry-point group `flex_orchestrator.plugins`, `@command(params=PydanticModel)`, `orch.store_data` / `get_data`, events. **The `StudyConfig` sweep union is closed.** `inputs/resolve.py` lines stored data up with a model's time steps (time zone, `offset`/`data_years`, interval resampling, gap rules, `reduce`, several time blocks); `fitting/prep.py` prepares fit data; `flexpse/build.py` turns values into `fix` elements | These data functions **move to flex-signal** in M3 and flex-orchestrator calls them (D22, `FLEX_ORCHESTRATOR_OVERLAP.md`). flex-orchestrator keeps stores, studies, sweeps, regressor wrappers and Ray. We ship no plugin (D6) |
| Acquirium (alpha, 0.4.0a8; build against `upstream/main`) | Python client returns **polars**. Data is stored in UTC. Writes need `register_streams` and then `insert_timeseries_arrow`. Derived streams are `aq.App` subclasses running **inside the server**; their module digest is pinned, and outputs must contain no nulls. There are no quality-flag columns and only one value per `(stream, ts)` | Use pandas internally and convert at the boundary. Ship one stable `PipelineApp` class whose behaviour comes entirely from parameters (the spec). Write flags as separate streams. Store forecast issuances as experiment runs or as per-horizon streams. |
| flows-prep (legacy, py<3.12) | Tukey and rate-of-change outlier removal, stuck-value detection, short-gap interpolation with a consecutive-NaN limit, lagged ElasticNet imputation, Fourier features, blocked CV, clipping to bounds. PyPOTS was planned but never committed | This is the method inventory to port as thin steps. Keep its config ideas, not its code. |
| flows-mpc (legacy) | statsforecast MSTL, statsmodels, a Keras seq2seq model, MLflow tracking | Confirms Nixtla and statsmodels as the forecasting backbone. Drop TensorFlow and use torch-based extras. |

**No data from other repositories, with one exception.** These repos inform the design only. flex-signal's tests,
examples, benchmarks and docs use data generated inside this repo, plus flex-pse's public test dataset (see §8,
"Test data").

**Python version.** flex-pse needs ≥3.11 and pandas ≥3. flex-orchestrator needs ≥3.12. Acquirium needs ≥3.11.
Python 3.10 reached end of life in October 2026, and darts, pint and sklearn 1.8 have already dropped it.
→ **`requires-python >=3.11`**, with CI on 3.11–3.14, matching flex-pse.

---

## 3. Architecture

### 3.1 Package layout

The import name is `flexsignal`, which matches flex-pse's `flexcore`/`flexops`/`flexparameterize`. The
distribution name is `flex-signal`. The repo uses a src layout with tests next to the code
(`src/flexsignal/<pkg>/tests/`), as flex-pse does.

```
src/flexsignal/
  core/            data container, quality flags, units metadata, exceptions, step/backend registries
  spec/            SignalSpec schema, load/dump, migrations, committed JSON Schema
  steps/
    select.py      align, resample, window, mask/condition, rename, combine
    clean.py       range, rate_of_change, hampel, tukey, stuck, expression, filter (savgol/butter/median), changepoint
    correct.py     conditional calibration / residual correction (the A-by-B case)
    impute.py      interpolate, kalman, regression, pypots, tiered
    forecast.py    forecast step + backtest
    derive.py      expression / function soft sensors (new streams from existing ones)
  backends/        lazy-imported adapters: sklearn, statsmodels, scipy, pypots, skforecast, nixtla, darts, sktime, chronos
  pipeline.py      Pipeline / FittedPipeline (fit, transform, forecast, save, load)
  io/              parquet (+ arrow metadata), csv
  eval/            masking benchmark for imputers, fit and forecast metrics
  handoff/         data to models: to_grid (values per model time step), fit_data (X, y for a regressor),
                   time-scope path rules; no model library imported (M3, D22)
  integrations/
    flexpse/       time_grid of a spec or model, apply_values to a spec or model (fix elements); flex-pse lazily
    acquirium/     source, sink, PipelineApp (last phase)
  api.py           run / fit / backtest entry points: the orchestrator-ready contract (§4.3)
```

import-linter layering (same tool as flex-pse):

```
core <- {spec | io} <- backends <- steps <- pipeline <- {eval | api | testing | handoff} <- integrations.{flexpse | acquirium}
```

Each `integrations.*` package imports its third-party target only inside itself. Each is behind its own extra.

**Import direction with flex-pse.** flex-pse workflows import flex-signal, so flex-signal must never import flex-pse
at module import time. This rules out circular imports in either direction. import-linter enforces two rules:

- no module outside `integrations.flexpse` may import `flexcore`, `flexops`, `flexparameterize`, `flexdesign`,
  `pyomo` or `idaes`;
- `integrations.flexpse` itself imports flex-pse only lazily, inside functions: `flexops` to read a spec's time
  element, find a built model's time blocks and apply `fix` elements. It doesn't import `flexparameterize`:
  flex-signal stops at a prepared `X, y` (D22).

`import flexsignal` therefore never pulls in pyomo or idaes.

### 3.2 Data model: `SignalFrame`

We do **not** create a new timeseries library. A `SignalFrame` is a thin wrapper around plain pandas objects:

- `values`: `pd.DataFrame`, tz-aware `DatetimeIndex` (UTC internally), one float column per stream.
- `flags`: `pd.DataFrame` of the same shape, `uint16` bitmask per sample.
- `meta`: `dict[str, StreamMeta]` holding units (pyomo/pint-parseable string, e.g. `"m^3/hr"`), quantity kind, source
  reference (Acquirium `ref_uri` or a file), and a description.
- `history`: an ordered list of step records (index, kind, name, spec hash, rows before and after, streams added
  and removed, samples changed per stream). This is the provenance. The pipeline writes it by comparing each
  step's input and output, so steps don't record themselves and the record is always exact.

Quality flags are an `IntFlag` bitmask:

```
RAW=0  MISSING  OUT_OF_RANGE  RATE_OF_CHANGE  OUTLIER  STUCK  MANUAL  EXCLUDED
CORRECTED  IMPUTED_SHORT  IMPUTED_MODEL  FORECAST  DERIVED  RESAMPLED  NO_SAMPLE
```

`EXCLUDED` (added in M1) marks data a selection rule left out on purpose, e.g. "only while the pump runs". It is
not a quality judgement, and imputers never fill it.

`NO_SAMPLE` (added in M1, D13) marks the rows a stream gets, when several inputs are joined, only because another
input has a sample at that time. It is always set together with `MISSING`. `align`, `resample` and `interpolate`
skip these rows, so an off-grid source aligns to its own samples, not to the join's padding.

A cleaning step does not delete data. It sets the value to NaN and sets a flag bit. An imputer fills NaNs and
sets `IMPUTED_*`. A downstream consumer can therefore always ask which values are real.

Steps receive and return a `SignalFrame`. `sf.df` / `sf.to_pandas()` give the bare DataFrame for any library
call. The Acquirium adapter converts polars to and from this frame at the boundary.

### 3.3 Step protocol

Steps follow scikit-learn's fit/transform split, made functional (settled in M1, D9):

```python
class StepSpec(StrictModel):            # frozen pydantic model: the step *is* its spec
    kind: str
    name: str | None
    def fit(self, sf: SignalFrame) -> dict: ...                         # JSON-able state; {} if stateless
    def transform(self, sf: SignalFrame, state: dict) -> SignalFrame: ...

class Forecaster(StepSpec):             # M4
    def predict(self, sf, state, horizon, *, exog_future=None) -> SignalFrame: ...
```

The step object holds configuration only, and fitted state is an explicit JSON dict. A Python-built and a
JSON-loaded pipeline are then the same objects and give bit-identical output. Fitting never changes the spec,
and a fitted pipeline saves as JSON and pickles for Ray. `Pipeline.fit` / `FittedPipeline.transform` give the
familiar sklearn feel at the pipeline level.

- **Backends** are registered by a string key, for example `"sklearn:HuberRegressor"`, `"statsmodels:SARIMAX"`,
  `"pypots:SAITS"`, `"nixtla:AutoETS"` or `"chronos:chronos-2"`. User code adds its own methods by registering them
  (§3.8); a spec never names an import path (D17, D21). Each key maps
  to a small adapter that turns a `SignalFrame` into the shape the library wants and back again, keeping the
  index. Examples:
  - PyPOTS needs `{"X": [n, T, F]}` windows.
  - Nixtla needs long format (`unique_id, ds, y`).
  - darts needs a `TimeSeries`.
- Third-party plugins add backends through a `flexsignal.backends` entry-point group, mirroring flex-pse's
  `flexpse.extensions` group.
- Heavy backends are imported lazily. A missing extra raises
  `FlexConfigError("backend 'pypots:SAITS' needs `pip install flex-signal[pypots]`")`.

### 3.4 Conditions and selection

Conditions use a small expression language that compiles to a boolean mask, with Python syntax over stream
names: `"C == 0 and D == 0 and B > 0"`. `check_condition` parses it with `ast` and allows only stream names,
numbers, arithmetic, comparisons, boolean operators and a fixed list of math functions, so a condition can't run
code (D15). Since M2, flex-signal also evaluates the parsed tree itself with numpy, so `pandas.eval` is no longer
involved (D16). New helpers must be added to the allowlist explicitly. On top of that it adds (M2):

- a duration helper: `held(C == 0, "30min")` means the condition has been true, without a break, for at least
  30 minutes;
- flag and value predicates: `valid(B)`, `is_raw(B)`, `not flagged(A, "OUTLIER")`.

Conditions are type-checked when a spec loads (a condition must give True/False; `and`/`or`/`not` need
True/False operands). The same language without the helpers gives **formulas**, numeric expressions with fitted
parameters, e.g. `"a + b * A"` for `scipy:curve_fit`.

Every step that fits or applies something accepts `fit_when` and `apply_when`. This is how the multi-stream
example is expressed without custom code.

### 3.5 Worked example: correcting A using B under conditions

B is a **trusted but intermittent** measurement of the same quantity as A. A is always available but has a
variable error. The `calibrate` step works like this:

1. **Fit.** It fits a mapping `A → B` on rows where `fit_when` holds and both A and B are valid. Here,
   `fit_when` is C == 0, D == 0 and B > 0.
2. **Apply.** It applies that mapping to A everywhere to produce `A_clean`.
3. **Prefer the reference.** With `prefer_reference=True`, `A_clean` takes B's own value wherever B is valid and
   trusted. Elsewhere it uses the calibrated A, and those values are flagged `CORRECTED`.
4. **Track drift.** With `refit="rolling"`, the calibration is refit over a moving window of trusted B samples,
   using the same model on each window. The calibration then follows drift between reference periods. (M2 refits
   per block of time, `every`, on the window ending with that block; windows with too few trusted rows fall back
   to the model fitted on the training data.)

`report()` covers how many samples were fit, residual statistics, and for each sample whether it was referenced,
calibrated or still missing.

```python
import flexsignal as fs

pipe = fs.Pipeline(name="stream_a_cleanup", steps=[
    fs.select.Align(freq="1min", how="inner"),                    # match B to A by time index
    fs.clean.Range("B", min=0),                                    # basic hygiene first
    fs.clean.Hampel(["A", "B"], window="15min", n_sigma=3),
    fs.correct.Calibrate(
        target="A",                         # stream with the variable error
        reference="B",                      # trusted signal while the condition holds
        features=["A"],                     # model input(s); add others if the error depends on them
        fit_when="C == 0 and D == 0 and B > 0",
        apply_when=None,                    # None = apply everywhere; elsewhere A passes through unchanged
        model="sklearn:HuberRegressor",     # or "statsmodels:RLM", or "scipy:curve_fit" with formula=/initial=
        prefer_reference=True,              # use B itself on the trusted rows; calibrated A elsewhere
        refit="static",                     # or {"rolling": "2D"} to follow drift between reference periods
        output="A_clean",                   # the default, <target>_clean; output="A" replaces A
    ),
    fs.impute.Tiered("A_clean", tiers=[
        {"max_gap": "15min", "method": "interpolate"},
        {"max_gap": None, "method": "kalman"},
    ]),
])

fitted = pipe.fit(sf_train)
clean = fitted.transform(sf)            # SignalFrame with A_clean + flags (CORRECTED / IMPUTED_*)
fitted.save("artifacts/stream_a_cleanup/")   # spec.json + fitted params (coefficients as JSON)
fitted.report()                         # fit metrics on fit_when rows, coverage, flag counts
```

The same pipeline as a spec. Python and JSON/YAML are interchangeable, both directions:

```json
{
  "schema_version": "0.1.0",
  "name": "stream_a_cleanup",
  "inputs": [{"kind": "parquet", "path": "raw.parquet"}],
  "steps": [
    {"kind": "align", "freq": "1min", "how": "inner"},
    {"kind": "range", "streams": ["B"], "min": 0},
    {"kind": "hampel", "streams": ["A", "B"], "window": "15min", "n_sigma": 3},
    {"kind": "calibrate", "target": "A", "reference": "B", "features": ["A"],
     "fit_when": "C == 0 and D == 0 and B > 0", "model": "sklearn:HuberRegressor",
     "prefer_reference": true, "refit": "static", "output": "A_clean"},
    {"kind": "tiered", "streams": ["A_clean"], "tiers": [
      {"max_gap": "15min", "method": "interpolate"}, {"max_gap": null, "method": "kalman"}]}
  ],
  "outputs": [{"kind": "parquet", "path": "clean.parquet"}]
}
```

### 3.6 Spec system (G4)

The spec mirrors `flexcore.config` one for one:

- `SignalSpec(_StrictModel)` has the fields `schema_version`, `name`, `inputs`, `steps`, `outputs`, `forecast`
  (optional) and `metadata`.
- `steps` is `list[Annotated[Union[...], Field(discriminator="kind")]]`. Plugin step kinds are added at
  registration time and the union is rebuilt.
- `load_spec(path|dict)` / `dump_spec(spec, path)` run migrations through a `MIGRATIONS` dict and raise
  `FlexConfigError` with dotted field paths.
- `export_json_schemas()` writes `src/flexsignal/spec/schemas/signal_spec.schema.json`, which is committed.
- `$source` refs point to large tables such as lookup curves or manual masks. `Quantity` is used for thresholds
  that carry units.
- Fitted state lives **next to** the spec, never inside it. A fitted pipeline is a directory: `spec.json`,
  `fitted/<step>.json` (coefficients, `popt`, equation strings), and `fitted/<step>.<native>` for torch or darts
  models. We avoid pickle, and use `skops` where sklearn objects can't be reduced to plain parameters.
- `spec_hash` is a sha256 of the canonical JSON. It is written into provenance in flags/history, Acquirium
  metadata and orchestrator datasets.

### 3.7 Forecasting

- Forecast steps (`fs.forecast.Structural`, `Regression`, `SeasonalNaive`, `Persistence`) take the target
  `streams`, `horizon`, `issue_time`, `exog` (streams known ahead) and `quantiles`, plus their method's options.
  A forecast step writes the future rows of the frame (D25): it runs from a JSON spec after the cleaning steps, and
  values known ahead are rows after the issue time in the input.
- `fs.backtest(spec, data, folds=..., compare=[...])` → scores per method and fold. It uses blocked, time-ordered
  folds, the same idea as flows-prep's `get_train_test_folds`, and hides everything except `exog` after each issue
  time. Its output feeds flex-pse model selection.
- The output is a `SignalFrame` whose forecast rows are flagged `FORECAST`; the target's metadata carries
  `issue_time`. Optional quantile columns are named `<stream>__q10` and so on.
- A forecast reaches a flex-pse model as fixed values, through `handoff.to_grid` and `apply_values` (D24).

### 3.8 Your own methods (extension API, M5)

Users building their own flex-signal workflow need to plug in their own cleaning, imputation, forecasting and
correction methods and run them from a JSON spec, without changing flex-signal's source. The rule that keeps this
safe (D21):

> **Only registered methods can run.** A method is registered by code: the user's script, or an installed
> package. A spec refers to it by name only. A spec never names an import path, and loading or running a spec
> never imports anything.

So a spec from someone else can only use methods the person running it has chosen to make available.

**Registering a method.** One function per role turns a plain Python function into a full step:

```python
import flexsignal as fs

def despike(values, *, window: str = "1h", k: float = 4.0):
    """Return True where a value should be rejected."""
    ...

def fill_from_neighbours(values, gaps, state, *, features: list[str], weight: float = 1.0):
    """Return proposed values; only the positions in `gaps` are used."""
    ...

fs.register_cleaner("PlantDespike", despike, flag="OUTLIER")
fs.register_imputer("CustomMethod", fill_from_neighbours)

pipe = fs.Pipeline([
    fs.clean.PlantDespike("flow", window="30min"),
    fs.impute.CustomMethod("flow", features=["pressure"], weight=0.5, max_gap="6h"),
])
```

and in a spec, once the same functions are registered:

```json
{"kind": "PlantDespike", "streams": ["flow"], "window": "30min"},
{"kind": "CustomMethod", "streams": ["flow"], "features": ["pressure"], "weight": 0.5, "max_gap": "6h"}
```

| Function | The user's function does | flex-signal does |
|---|---|---|
| `register_cleaner(name, fn, flag=...)` | returns which values to reject | sets them to NaN, adds the flag, skips values already missing |
| `register_imputer(name, fn, fit=None)` | proposes values for the gaps it is given | finds the gaps (`max_gap`, never `EXCLUDED` or `NO_SAMPLE`), writes only into them, flags `IMPUTED_MODEL`; usable as a `tiered` tier by name |
| `register_forecaster(name, fn, fit=None)` | returns values for the horizon, from history and known future inputs | the M4 `Forecaster` protocol: issue time, `FORECAST` flag, quantile columns, backtesting |
| `register_model(name, cls)` | a scikit-learn-style regressor whose fitted `coef_`/`intercept_` describe it | makes it a `model` choice for `calibrate` and `regression`; the fitted state stays JSON |

- **Options come from the function's signature.** Keyword-only parameters, with their type hints and defaults,
  become the step's fields through a generated pydantic model. JSON options are validated like any built-in step's,
  and errors name the field (`steps.3.weight: ...`). The standard fields (`streams`, `name`, and `max_gap` for
  imputers) are added by flex-signal.
- **Fitting.** An optional `fit(values, **options) -> dict` learns from training data. Its result must be JSON, as
  for built-in steps (D9), so fitted pipelines still save without pickle.
- **Names.** A registered name must be a Python identifier and must not clash with a built-in kind. It is the
  step's `kind` in JSON and its name in `fs.clean` / `fs.impute` / `fs.forecast` (resolved by each module's
  `__getattr__`). Packages shared between teams should prefix their names, e.g. `PlantDespike`.
- **Unregistered methods are errors**, raised when the spec loads: *unknown step kind 'CustomMethod'; register it
  first, e.g. fs.register_imputer('CustomMethod', fn), or install the package that provides it*.

**Making methods available without editing flex-signal.** Three ways, all chosen by the person running the spec:

1. Register in the script, before loading the spec (as above).
2. Install a package that registers its methods through the existing `flexsignal.steps` entry-point group; they
   are available as soon as the package is installed.
3. Name the modules to load in the call: `fs.load_plugins("myplant.methods")`, or
   `fs.run(spec, data, plugins=["myplant.methods"])`. The module list comes from the caller's code, never from the
   spec. A spec may list what it needs under `"requires": ["myplant.methods"]`, but only so the error message can
   say what to install; flex-signal never imports from it.

**Provenance and reuse.** Each history record of a custom step names the function behind it (module, qualified
name, and the providing package's version), so a result says which code produced it. Loading a saved fitted
pipeline needs the same methods registered first. For Ray workers, register from an importable module (option 2
or 3) so workers can register the same methods.

**One path for everyone.** `register_step` with a `StepSpec` subclass stays as the low-level route for full
control. M5's own adapters (PyPOTS, Chronos, pyod, ...) are registered through the same public functions, so the
extension API is exercised by flex-signal itself. The `function` step of the Derive category (§6) becomes a
registered function looked up by name, not an import path in the spec.

---

## 4. Ecosystem integration

### 4.1 flex-pse and the hand-off to models (always the latest flex-pse release; currently 0.3.1)

flex-signal, flex-pse and flex-orchestrator are used together, but everything must also work **in a plain script
without flex-orchestrator**. So the data work of handing timeseries to a model lives in flex-signal, and
flex-orchestrator calls it (D22). Until M3 these functions live in flex-orchestrator (`inputs/resolve.py`,
`inputs/scopes.py`, `fitting/prep.py`, `flexpse/timegrid.py`, `flexpse/build.py`). M3 moves them here with their
tests and behaviour unchanged. `FLEX_ORCHESTRATOR_OVERLAP.md` is the item-by-item inventory and migration plan.

Two packages, split by whether they need a model library:

- **`flexsignal.handoff`** (pure data, no flex-pse import):
  - `to_grid(data, grid, options) -> GridValues`: one value per model time step. It converts UTC to the
    facility's wall clock and drops the zone, reads another window (`offset`) or another year's dates
    (`data_years`), resamples `forward` `[t, t+dt)` or `backward` `(t-dt, t]` with an `agg`, splits coarser
    summed amounts over whole steps, and applies a gap rule (`interpolate` / `ffill` / `zero` / `error`, with a
    `limit`), or `reduce`s the horizon to one number. `GridValues` carries the values, units (`^`, not `**`),
    how many steps were filled, a data hash, and `to_overrides(rename=...)`. With a `SignalFrame`, samples
    flagged `EXCLUDED` or `NO_SAMPLE` aren't samples, and steps resting on filled or corrected values are
    reported.
  - `data_window(grid, options)`: the UTC window to read for a horizon, with the margin end gaps need.
  - `GridOptions`: those options as a pydantic model, with flex-orchestrator's field names and defaults.
  - `grids_of`, `scope_of`, `targets`: which time block a model path belongs to, for models with several.
  - `fit_data(X, y, prep) -> FitData`, `split_holdout`, `data_hash`, `FitPrep`: join input and output frames,
    resample, apply a gap rule (`error` / `drop` / `interpolate` / `longest_segment`), guard the row count,
    and hash the result. **flex-signal stops here**: a script passes `X, y` to a flex-pse regressor itself.
    flex-orchestrator keeps its regressor wrappers for sweeps and automated fits.
- **`flexsignal.integrations.flexpse`** (imports flex-pse lazily):
  - `time_grid(spec_or_model)`: the time points of a flowsheet spec's time element, or of a built model's time
    blocks, one axis per scope. Design specs stay in flex-orchestrator, which calls this per horizon.
  - `apply_values(spec_or_model, values)`: the same values on either, with the same result. On a flowsheet spec,
    a copy with one named `fix` element per value; a path into a `costing` element edits that field (how a flat
    price is set). On a built model, in place: mutable Params (a price built as a Param) are set, and the rest are
    fixed, grouped by time block, via `flexops.apply_spec`.

```python
from flexsignal import handoff
from flexsignal.integrations import flexpse as fsp

clean = fs.run(spec, raw).signal                                   # clean, calibrate, fill
grid = fsp.time_grid(model)
values = handoff.to_grid(clean, grid, handoff.GridOptions(timezone="America/Los_Angeles", gaps="error"))
fsp.apply_values(model, values.to_overrides(rename={"A_clean": "plant.feed.flow_vol"}))

fit = handoff.fit_data(clean.select(["flow"]), clean.select(["power"]), handoff.FitPrep(resample={"freq": "1h"}))
regressor = LinearRegressor().fit(fit.X, fit.y)                    # flex-pse, called directly
```

`arima_history(...)`, for seeding an ARIMA surrogate, waits for a flex-pse release with the ARIMA surrogate. It is
on flex-pse `main` (merged 2026-11-10) but not in 0.3.1.

### 4.2 Acquirium (`flex-signal[acquirium]`, last phase)

We build against acquirium `main`, and the integration is scheduled last (phase 7) so that Acquirium's alpha API
can settle first. Until then, nothing else in flex-signal depends on Acquirium: data arrives as DataFrames or
Parquet. The design below is provisional and will be re-checked against acquirium `main` when phase 7 starts.

- **Source:** `AcquiriumSource(query=..., start, end, streams=[(source_id, ref_name)|ref_uri])` → `SignalFrame`.
  `meta` is filled from the point metadata: unit, quantity kind and `ref_uri`. Polars is converted to pandas, and
  times stay UTC.
- **Sink:** `AcquiriumSink(source_id="flex-signal:<pipeline>")` registers streams that carry the input's unit and
  quantity kind, with metadata `{spec_hash, step, flags_stream}`. It then calls `insert_timeseries_arrow`.
  - Flags are written as companion numeric streams (`<name>__flags`), because Acquirium has no flag column.
  - Unfilled NaNs are dropped before writing.
- **Live soft sensors and cleaning:** a single stable class, `flexsignal.integrations.acquirium.app:PipelineApp`.
  Its behaviour comes from `parameters={"spec": <SignalSpec json>, "fitted": <artifact ref>}`.
  - Because the module stays the same, the module digest Acquirium pins stays the same, and changing the spec
    means redeploying parameters, not code.
  - Requires flex-signal installed in the Acquirium server environment.
  - Steps must be deterministic and stateless.
  - `lookback` is derived from the largest window any step needs.
- **Forecasts:** each forecast issuance is stored as an Acquirium experiment run (`study.output(...).timeseries`).
  Two optional extras: a rolling "latest forecast" stream per horizon, and fitted models stored through
  Acquirium's `object` codec, which already handles flex-pse models.

### 4.3 flex-orchestrator: an orchestrator-ready API

flex-signal ships **no orchestrator code**. flex-orchestrator depends on flex-signal: from M3 it calls
`flexsignal.handoff` and `flexsignal.integrations.flexpse` for `model_inputs` and fit preparation (D22,
`FLEX_ORCHESTRATOR_OVERLAP.md`), and once flex-signal is stable it adds a wrapper for running specs, as commands
and/or a `signals:` stage in `StudyConfig`. Our job is to make both trivial. The public API therefore follows this contract, which matches how the orchestrator already calls
flex-pse and its regressors:

| Contract | What it means for our API |
|---|---|
| **Pure functions with a spec in and JSON-able results out** | `flexsignal.run(spec, data) -> RunResult`, `flexsignal.fit(spec, data) -> FittedPipeline`, `fitted.transform(data)` (a forecast step in it issues from the newest data), `flexsignal.backtest(spec, data, folds=)`. `spec` may be a `SignalSpec`, a dict or a path. These map one-to-one onto orchestrator `@command`s whose params model simply embeds `SignalSpec` |
| **No I/O inside the pipeline when data is passed in** | Spec `inputs` are *named slots*, e.g. `{"kind": "frame", "name": "historian"}`. The caller binds them: `data={"historian": (df, units)}`. Orchestrator `Selector`s resolve to these slots. File and Acquirium inputs exist only for standalone use |
| **The orchestrator's data shape** | Data goes in and out as a `pd.DataFrame` with a tz-aware UTC `DatetimeIndex` plus `units: dict[str, str]`, exactly what `orch.get_data` / `orch.store_data(name, df, units=)` use. `RunResult` exposes `.data`, `.units`, `.flags` (DataFrame), `.report` (JSON-able dict), `.spec_hash`, `.provenance` |
| **Storage-agnostic fitted state** | `FittedPipeline.to_payload() -> {"spec": dict, "state": dict, "blobs": dict[str, bytes]}` and `FittedPipeline.from_payload(...)`. Directory save/load is built on top of this. The orchestrator can then store fits in any backend (file, memory, Acquirium) |
| **Ray-safe** | No global mutable state. Fitted pipelines are picklable. Random seeds are explicit in the spec. Backends are imported lazily on workers |
| **Progress and errors** | An optional `progress: Callable[[float, str], None]` that the orchestrator's `Progress` adapts to. All errors subclass `SignalError` with `field=` dotted paths, so the command ledger records something actionable |
| **Embeddable schema** | `SignalSpec` and its JSON Schema are exported with stable `kind` names and `schema_version` migrations, so `StudyConfig` can embed a `signals:` list without redefining it |
| **Time semantics** | UTC internally. Naive input timestamps are taken as the machine's local time with a `NaiveTimestampWarning` (D14), so the orchestrator should always pass tz-aware data or set `timezone`. Local naive time is emitted only by `handoff.to_grid` (`GridOptions.timezone`, default `"UTC"` as in the orchestrator), which is the orchestrator's own `model_inputs` alignment once it moves here (D22) |

**Verification:** `tests/contract/` exercises this contract without importing flex-orchestrator. It wraps the
API in a pydantic params model, JSON round-trips specs, results and payloads, and pickles a fitted pipeline. An
optional integration test runs against flex-orchestrator's memory backend when it is installed. The handoff
functions carry flex-orchestrator's own alignment and fit-prep tests, ported as pure-data tests (M3).

The orchestrator wrapper will also let us retire the orchestrator's own gap filling (`fitting/prep.py`,
`inputs/resolve.py`). That change is up to the flex-orchestrator side.

---

## 5. Dependencies

### 5.1 Core (no torch, all wheels)

`pandas>=3` (matches flex-pse), `numpy`, `scipy`, `scikit-learn>=1.6`, `statsmodels>=0.14`, `pydantic>=2`, `pyarrow`.
`pandera` is optional for validating inputs and outputs: monotonic tz-aware index, unique timestamps, dtypes.
Default: include it.

### 5.2 Extras

| Extra | Packages | Purpose |
|---|---|---|
| `pypots` | pypots (+ pygrinder) | Deep imputation and forecasting (SAITS, BRITS, CSDI, TimesNet), and pygrinder's missing-data patterns for the benchmark. Pulls in torch and transformers. M2's benchmark hides block and random gaps itself, so there is no separate `eval` extra (D20) |
| `ml` | skforecast, lightgbm | Lightweight, pandas-native ML forecasting with exogenous variables. **On hold:** skforecast requires `pandas<3` (D23, O2) |
| `nixtla` | statsforecast, mlforecast | Fast AutoARIMA, ETS, MSTL and Theta. Successor to the flows-mpc stack. **On hold:** both require `pandas<3` (D23, O2) |
| `deep` | neuralforecast | NHITS, NBEATS(x), MLP, LSTM, TiDE, PatchTST, DLinear (M5) |
| `foundation` | chronos-forecasting (Chronos-2); timesfm optional (2.5) | Zero-shot forecasting, Apache-2.0 weights only (D29) |
| `t0` | tfc-t0 | t0-alpha zero-shot forecasting (Apache-2.0). tfc-t0 needs huggingface-hub ≥1 and PyPOTS 2.0 needs <1, so `[t0]` can't be installed with `[pypots]` and isn't in `[all]`; CI tests it in its own job |
| `darts` | darts | Wide model zoo through an adapter. Heavy, and opt-in only |
| `sktime` | sktime | Adapter to the sktime estimator zoo |
| `anomaly` | pyod, ruptures, stumpy | ML outlier detectors, change points (regime and drift), matrix-profile discords. ruptures has no Python 3.14 wheels yet |
| `features` | tsfresh | Feature extraction for regression imputers |
| `units` | pint | Unit conversion where asked: `convert_units`, and `range` bounds in other units (M5). Pint 0.26 needs Python ≥3.12, so the floor is 0.24 |
| `symbolic` | pysr | Symbolic-regression correction models. Needs a Julia runtime |
| `flexpse` | flex-pse, **the latest release** (now 0.3.1) | Used by `integrations.flexpse` (`time_grid`, `apply_values`) and its tests. flex-pse is the importer, not a dependency of core. flex-pse is not on PyPI, and PyPI rejects git URLs in a published package's metadata. So until flex-pse is published, the pin lives in `environment.yml` and CI as `flex-pse[parameterize] @ git+https://github.com/flex-pse/flex-pse.git@<tag>`, the same way flex-orchestrator pins it. The tag's dist-info still says 0.1.0, so pin by tag, never by `==version`. Once flex-pse is on PyPI, the extra becomes `flex-pse[parameterize]>=<latest>` |
| `acquirium` | acquirium (tracks `main`) | Source, sink and PipelineApp (phase 7) |
| `dev` | pytest, pytest-cov, pytest-xdist, ruff, black, import-linter, pre-commit | Same pins as flex-pse |

### 5.3 Considered and skipped

| Package | Reason skipped |
|---|---|
| ADTK, Merlion | Unmaintained |
| Prophet | Heavy Stan dependency and a weak fit for sub-hourly process data. Still reachable through darts or sktime |
| GluonTS, uni2ts/Moirai | Redundant. Moirai's weights are licensed CC-BY-NC |
| Great Expectations | Too platform-like for a library |
| Hydra | Application config; users can layer it on top of our specs |
| TensorFlow | Being dropped in favour of the torch ecosystem |

### 5.4 Backbone decision

We considered building on sktime/skbase or darts as the backbone and decided against it. Instead we write our own
thin pandas-native protocol on sklearn's `BaseEstimator`, and treat sktime and darts as adapters. Reasons:

- It keeps core dependencies light.
- It avoids sktime's mtype churn and darts's non-pandas container.
- It still reaches both model zoos.

---

## 6. Method inventory (phase 1–2 targets)

| Category | Step `kind` | Backed by |
|---|---|---|
| Select | `align`, `resample`, `window`, `mask`, `rename`, `combine`, `shift` | pandas |
| Clean | `range`, `rate_of_change`, `stuck`, `hampel`, `tukey`, `expression` | pandas (in-house, ~15 lines each; ported from flows-prep) |
| Clean | `filter` (savgol, butterworth `sosfiltfilt`, median); sets `FILTERED` (D18) | scipy.signal / pandas |
| Clean | `changepoint` | ruptures `[anomaly]` |
| Clean | `detector` | pyod `[anomaly]` |
| Clean | `detotalize` (totalizer → rate) | pandas (from flows-prep) |
| Correct | `calibrate` (map/residual, `fit_when`/`apply_when`) | sklearn / statsmodels linear models, `scipy.optimize.curve_fit` on a checked formula (D17); pysr later under `[symbolic]` |
| Correct | `mass_balance` (reconcile children to parent) | scipy.optimize least squares |
| Impute | `interpolate` (linear/time/pchip/akima, `max_gap`) | pandas / scipy |
| Impute | `kalman` | statsmodels state space (`UnobservedComponents` / `SARIMAX` smoother) |
| Impute | `regression` (lags, Fourier, other streams) | sklearn (ElasticNet / GBM) |
| Impute | `knn`, `iterative` (fitted on the data they fill; D19) | sklearn |
| Impute | `pypots` | PyPOTS `[pypots]` |
| Impute | `tiered` | Composition of the methods above, routed by gap length |
| Forecast | `forecast`, `backtest` | statsmodels, skforecast, statsforecast, neuralforecast, chronos, pypots, darts/sktime adapters |
| Derive | `expression`, `function` | checked conditions (D15) / a registered function, looked up by name (§3.8, D21) |
| Any | user-registered cleaners, imputers, forecasters and models | the user's own code, registered before the spec loads (§3.8, D21) |

---

## 7. Documentation

The documentation mirrors flex-pse's setup (`docs/`, `.github/workflows/docs.yml`; first mirrored from v0.2.0 and kept in step with each flex-pse release, now 0.3.1), so the two projects
read as one ecosystem.

**Tooling**

- **Sphinx** with **furo**, **myst-parser** (pages in Markdown, `index.rst` as the root), `autodoc`, `autosummary`
  (`autosummary_generate = True`), `napoleon` (Google docstrings only) and `intersphinx`.
- **Intersphinx targets:** python, pandas, numpy, scipy, scikit-learn, statsmodels, pydantic and flex-pse
  (`https://flex-pse.github.io/flex-pse/`). We link to flex-pse's `TimeBlock`, `apply_spec` and
  `flexparameterize` pages instead of re-describing them.
- **Strict builds.** We use `nitpicky = True` and `sphinx-build -W --keep-going`, so a broken cross-reference
  fails CI. Any `nitpick_ignore` entries are commented one by one, as in flex-pse.
- **Docstrings.** Google style is enforced by ruff's `D` rules with `pydocstyle.convention = "google"`. This goes
  one step further than flex-pse, which does not select `D`, and matches flex-orchestrator. Every public function,
  class and pydantic field has a docstring or `description`. Field descriptions go verbatim into the JSON Schema,
  so they are plain text.
- **Heavy extras are mocked.** Optional backends (torch, pypots, darts, chronos, ...) are listed in
  `autodoc_mock_imports`, so the docs build needs only `.[docs]`. flex-pse is imported lazily in our code, so the
  docs build does not need it either. Intersphinx covers the links.
- **Same skip hook as flex-pse.** Members marked `.. deprecated::` or `TODO` are dropped from the generated reference.
- **Deployment.** `docs.yml` builds on every PR and deploys to GitHub Pages from `main` (upload-pages-artifact,
  deploy-pages), on Python 3.13.

**Generated from code, so it cannot drift.** `docs/_ext/flexsignaldoc.py` is the counterpart of flex-pse's
`flexdoc.py`. Its collection functions are plain Python and are unit-tested directly. It provides these directives:

| Directive | Renders |
|---|---|
| `.. flexsignal-step-table::` | Every registered step `kind`: category, backing library, required extra, one-line summary |
| `.. flexsignal-spec-table:: <pydantic model>` | Fields of a step or spec model: name, type, default, description |
| `.. flexsignal-backend-table::` | Every backend key (`sklearn:*`, `pypots:*`, ...) with its extra and a link to the upstream docs |
| `.. flexsignal-flag-table::` | The quality-flag bits and what sets each one |

The committed `signal_spec.schema.json` is also published as a downloadable page.

**Layout** (Diátaxis, same sections as flex-pse)

```
docs/
  index.rst
  getting_started/   installation (core + extras matrix), ten_minutes (the A/B/C/D example end to end)
  explanation/       signal_frame_and_flags, spec_schema, conditions, steps_and_backends, calibration,
                     imputation_tiers, forecasting_and_backtesting, time_and_units, flexpse_integration,
                     orchestrator_contract, acquirium_integration (M7)
  how_to/            clean_a_stream, calibrate_against_a_reference, impute_gaps, evaluate_imputers,
                     forecast_and_backtest, prepare_fit_data, apply_to_a_flexpse_model,
                     save_and_load_a_fitted_pipeline, add_your_own_method, deploy_a_pipeline_app (M7)
  tutorials/         index page linking the exported marimo tutorials (D12); the notebooks live in /tutorials
  examples/          runnable scripts on flexsignal.testing.synthetic data and the vendored flex-pse biogas data
  reference/         flexsignal.core, spec, steps (generated tables), backends, pipeline, api,
                     integrations.flexpse, glossary
  changelog.md       includes CHANGELOG.md
  _ext/ _static/ _templates/autosummary/
```

**Examples are tested.** Every script in `docs/examples/` and every how-to code block runs in CI under the
`integration` marker. Examples that need an optional extra are skipped when it isn't installed.

**Tutorials (D12).** A numbered series of [marimo](https://marimo.io) notebooks in `tutorials/` teaches the API
from simple to complex. marimo is also what flex-pse-examples uses. Each milestone adds the tutorials for the
features it ships, in the same PR as the code, so no tutorial is written against an API that has since changed.

- **Files.** `tutorials/NN_short_name.py`, numbered without gaps (a unit test checks the order). They are plain
  `.py` marimo files, so they diff cleanly and ruff and black check them. Tutorials are exempt only from the `D`,
  `B018` (marimo displays a cell's last expression) and `E501` (markdown table rows) rules.
- **Data.** Only `flexsignal.testing.synthetic`, plus the vendored biogas data once it is in (see "Test data").
  Each tutorial injects faults at known places and scores every step against them: caught, missed, false alarms.
- **Shape.** Prose goes in `mo.md` cells, with one idea per section and a short "Next" at the end. Interactive
  controls (`mo.ui.slider`, `dropdown`, `switch`) appear wherever a parameter's effect is the lesson. Plots use
  matplotlib with a shared palette and `style(ax)` helper, defined in each notebook so it stands alone. Lines
  that exist only for the notebook (`mo.md`, `mo.vstack`, `mo.ui`, ...) carry a `# Display only (marimo)` or
  `# Notebook control (marimo)` comment, so readers can tell them from the flex-signal code they will copy.
  Tutorial 01 builds its `SignalFrame` from a plain DataFrame with `from_frame`; later tutorials use the
  `case.signal()` shortcut.
- **Tested.** The last code cell asserts the results the prose claims, for example "removed exactly the injected
  faults". The `integration` tier runs every tutorial as a script (`python tutorials/NN_x.py`) with the
  `[tutorials]` extra, so a wrong claim fails CI, not just a crash.
- **Published.** `docs.yml` runs `marimo export html` on each tutorial, and the export fails if a cell raises.
  `conf.py` copies the static pages into the site next to `docs/tutorials/index.md`. The static pages show code
  and results. The controls only work when a tutorial is run locally (`marimo edit`).
- **Kept here, not in flex-pse-examples.** flex-pse-examples holds applied case studies and tracks flex-pse
  `main`. These tutorials teach this package's API and must change with it. A cross-package case study goes to
  flex-pse-examples once the flex-pse bridge exists (M3 or later).
- **Browser-run pages (later).** Pyodide ships pandas 3, scipy, scikit-learn, statsmodels and pyarrow, so the
  cleaning tutorials could run live in the browser (`marimo export html-wasm`). That needs two things:
  flex-signal must be installable there (on PyPI from M6, or a wheel served with the site), and `pandera` must
  be dropped from the core dependencies or made optional, because Pyodide doesn't ship it and nothing imports it
  yet. Tutorials whose steps can't run in a browser (pyomo, torch) would keep static pages, or use precomputed
  results as flex-pse-examples does.

**Writing standard.** We follow flex-pse's `CONTRIBUTING.md`: plain, direct prose written for a human reviewer,
and error messages that say what went wrong and what to do. We also mirror its GitHub issue templates (including
"documentation improvement") and its PR template.

**Definition of done.** Every phase ships with its explanation, how-to and reference pages, and with a `-W` docs
build that passes. A feature without docs is not finished.

---

## 8. Milestones

Each milestone is a tagged release with a CHANGELOG entry. Milestone *Mn* is the same thing as "phase n"
elsewhere in this document.

The versions follow semver. During 0.x the API and `schema_version` may change, and every spec change ships with
a migration. 1.0.0 is the stable hand-off to flex-orchestrator. Acquirium comes after that, as 1.1.0 (D7).

```
M0 ─► M1 ─┬─► M2 ─┬─► M4 ─► M5 ─► M6 (1.0) ─► M7 (Acquirium)
          └─► M3 ─┘   (M3 can run alongside M2)
```

| Milestone | Version | Theme | Headline acceptance test | Depends on |
|---|---|---|---|---|
| M0 | 0.0.1 | Scaffold | CI, lint, import-linter and `sphinx-build -W` green on an empty package; Pages site live | — |
| M1 | 0.1.0 | Core, spec and cleaning | A cleaning pipeline gives identical output from Python and from JSON, and catches every injected fault in the synthetic fixtures | M0 |
| M2 | 0.2.0 | Calibrate and impute | The A/B/C/D example recovers the true A within tolerance; the imputer benchmark runs on synthetic and biogas data | M1 |
| M3 | 0.3.0 | Hand-off to models | flex-orchestrator's alignment and fit-prep tests pass against `flexsignal.handoff`; a plain script runs raw data → clean → `to_grid` → `apply_values` → solve, and `fit_data` → a flex-pse regressor, on the latest flex-pse release (0.3.1) | M1 (M2 for the full demo) |
| M4 | 0.4.0 | Forecasting | Backtest on synthetic data with known seasonality beats a seasonal-naive baseline; the forecast reaches a model as fixed values | M2, M3 |
| M5 | 0.5.0 | Advanced backends | PyPOTS, Chronos, anomaly and darts/sktime adapters pass their tests, skipped when an extra is missing | M2, M4 |
| M6 | 1.0.0 | Stable release | Orchestrator contract suite green; schema frozen at 1.0.0; published to PyPI; hand-off | M1–M5 |
| M7 | 1.1.0 | Acquirium | A live `PipelineApp` runs on a local Acquirium server (`main`), fed synthetic data by an in-repo test driver | M6 |

Every milestone also needs, per the definition of done in §7:

- its docs pages, with `-W` passing;
- its tutorials (D12, §7), built in the same milestone: each runs as a script in CI, exports to a static page,
  and asserts what it claims;
- a CHANGELOG entry;
- every test carrying exactly one tier marker;
- import-linter green;
- a git tag.

### M0: Scaffold (0.0.1)

- **Scope:**
  - `pyproject.toml`: setuptools, src layout, `requires-python >=3.11`, the core dependencies, and
    `[testing]`, `[dev]` and `[docs]`. Other extras are declared in the milestone whose code uses them.
  - `src/flexsignal/__init__.py` with `__version__`.
  - ruff (`E,F,I,B,UP,D`, google), black 88, and `.importlinter` with the layer contract and the "no flex-pse
    outside `integrations.flexpse`" rule.
  - pre-commit with a pre-push hook; pytest markers, and a root `conftest.py` that requires exactly one tier
    marker per test.
  - `ci.yml` mirroring flex-pse's latest release (uv installs; 3.11–3.14 on Ubuntu, 3.13 on macOS and Windows;
    one coverage run with `codecov.yml`; a wheel-based standard-install check), `docs.yml` (GitHub Pages) and
    `publish.yml`.
  - `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE` (Apache-2.0, as flex-pse), issue and PR templates.
  - Docs skeleton: `conf.py`, the §7 layout with one index page per section listing its planned pages, and an
    empty `_ext/flexsignaldoc.py`.
- **Acceptance:** a fresh clone passes `pip install -e .[dev,docs]`, then `pre-commit run -a`, `pytest`,
  `lint-imports` and `sphinx-build -W`. The site is published.

### M1: Core, spec and cleaning (0.1.0)

- **Scope:**
  - `core`: `SignalFrame`, the `Flag` bitmask, `StreamMeta`, the `SignalError` hierarchy, and the step and
    backend registries with the entry-point group.
  - `spec`: `SignalSpec`, the discriminated step union, named input slots, `load_spec`/`dump_spec`, `MIGRATIONS`,
    `$source`, `Quantity`, `spec_hash`, and the committed JSON Schema.
  - `select` steps: `align`, `resample`, `window`, `mask`, `rename`, `combine`, `shift`.
  - `clean` steps: `range`, `rate_of_change`, `stuck`, `hampel`, `tukey`, `expression`.
  - `impute.interpolate` with `max_gap`.
  - Parquet I/O with units and flags in the Arrow metadata, plus CSV.
  - `Pipeline` and `FittedPipeline` (`fit`, `transform`, `to_payload`/`from_payload`, `save`/`load`, `report`).
  - `api.run` and `api.fit`.
  - `flexsignal.testing.synthetic`: a seeded generator of plant-like data with known injected faults (spikes,
    stuck values, drift, out-of-range values, gaps) and their ground truth.
  - Vendor the flex-pse biogas test data into `src/flexsignal/tests/data/flexpse/` with a provenance README,
    once its upstream licence is confirmed to allow redistribution (see "Test data" below).
- **Acceptance:**
  - Each cleaner flags exactly the injected faults in its deterministic unit test.
  - A Python-built and a JSON-loaded pipeline give bit-identical output.
  - `load_spec(dump_spec(s)) == s` for every step kind.
  - Contract tests: specs, results and payloads JSON round-trip, and a fitted pipeline pickles.
  - `import flexsignal` imports no torch, pyomo or flex-pse (checked in a test).
- **Docs:**
  - getting_started/installation;
  - explanation pages: signal_frame_and_flags, spec_schema, steps_and_backends, time_and_units;
  - how_to/clean_a_stream and how_to/save_and_load_a_fitted_pipeline;
  - generated step, spec and flag tables.
- **Tutorials:**
  - 01 A first look: `SignalFrame`, units, flags, bringing your own data, the synthetic faults;
  - 02 Cleaning one stream: each cleaner scored against the injected faults, Hampel tuning, gap filling, why
    step order matters;
  - 03 Pipelines and specs: JSON specs, `spec_hash`, provenance, validation errors, fitting `tukey` on
    training data, save/load and payloads, file inputs and outputs;
  - 04 Several streams: named inputs, `align`, finding and shifting out a delay, `mask` before cleaning,
    multi-stream `expression` rules, `combine`, `resample` and `window`.

### M2: Calibrate and impute (0.2.0)

- **Scope:**
  - The condition language: Python syntax plus `held(...)`, `valid(...)`, `is_raw(...)` and `flagged(...)`,
    evaluated by flex-signal itself (D16).
  - `calibrate` with `fit_when`/`apply_when`, `prefer_reference`, and `refit` static or rolling. Backends:
    sklearn, statsmodels and `curve_fit` (D17).
  - Imputers: `kalman` (statsmodels), `regression` (lags, Fourier terms, other streams), `knn`, `iterative` and
    `tiered`.
  - `filter` (savgol, butterworth, median) and `detotalize`.
  - `eval`: a built-in masking benchmark (random and block gaps). pygrinder moves to `[pypots]` in M5 (D20).
  - `synthetic.abcd`: the A/B/C/D fixture with known truth, optional drift, and a conditionally wrong reference.
- **Acceptance:**
  - On a synthetic A/B/C/D fixture with known true A and intermittent B, `A_clean` RMSE is under a set threshold.
  - With `refit="rolling"`, the calibration tracks an injected drift that `static` misses.
  - The imputer benchmark report runs on synthetic data and on the flex-pse biogas data.
  - An `interpolate` pipeline reproduces flex-pse's `imputed_bio_gas_generation.csv`.
  - The two biogas items wait on open question O1 (§9).
- **Docs:**
  - getting_started/ten_minutes, built around the A/B/C/D example;
  - explanation pages: conditions, calibration, imputation_tiers;
  - how-to guides: calibrate_against_a_reference, impute_gaps, evaluate_imputers.
- **Tutorials:**
  - 05 Calibrating A against an intermittent reference B, only when C and D are 0: the full A/B/C/D problem,
    picking up from tutorial 04;
  - 06 Comparing imputers by gap length, on synthetic and biogas data.

### M3: Hand-off to models (0.3.0)

Moves the data functions that flex-orchestrator and this plan both had, into flex-signal, so a plain script can
hand data to a flex-pse model and flex-orchestrator can depend on flex-signal (D22).

**Status (2026-11-15): built.** `flexsignal.handoff` and `flexsignal.integrations.flexpse` pass flex-orchestrator's
ported tests and the end-to-end test against flex-pse 0.3.1 with HiGHS; tutorial 07 and the docs are in. To tag
0.3.0 once merged; flex-orchestrator's switch follows in its own repo. The item-by-item inventory,
the behaviour to preserve and flex-orchestrator's side of the migration are in `FLEX_ORCHESTRATOR_OVERLAP.md`.

- **Scope:**
  - `flexsignal.handoff`: `GridOptions`, `data_window`, `to_grid` → `GridValues` (values, units, filled
    counts, data hash, `to_overrides`); `grids_of`, `scope_of`, `targets` for several time blocks; `FitPrep`,
    `fit_data` → `FitData`, `split_holdout`, `data_hash`.
  - `flexsignal.eval.metrics` (RMSE, MAE, MAPE).
  - `flexsignal.integrations.flexpse`: `time_grid` (flowsheet specs and built models, several time blocks),
    `time_scopes`, `apply_values` (on a spec or a built model).
  - Flag awareness: `EXCLUDED` and `NO_SAMPLE` samples aren't samples; `GridValues` reports steps resting on
    filled or corrected values.
  - CI job pinned to the latest flex-pse release by git tag (`flex-pse @ git+...@0.3.1` today; see D3), plus a
    scheduled job against flex-pse `main`. import-linter adds `flexdesign` to the flex-pse-only modules.
  - Not in scope: regressor wrappers (flex-signal stops at `X, y`), design-spec grids (stay in
    flex-orchestrator), store selectors, studies and sweeps.
- **Acceptance:**
  - flex-orchestrator's tests listed in `FLEX_ORCHESTRATOR_OVERLAP.md` §4 pass as pure-data tests against
    `flexsignal.handoff` / `integrations.flexpse`, with the same inputs, values, errors and data hashes.
  - A component test runs synthetic raw data → clean → `time_grid` → `to_grid` → `apply_values` on a model built
    with flex-pse 0.3.1, then solves with HiGHS; and `fit_data` → flex-pse's `LinearRegressor`.
  - Gaps a rule can't fill, off-grid data and an unknown path raise errors that name the column, the steps and
    the fix.
  - `import flexsignal` and `import flexsignal.handoff` still import no flex-pse, pyomo or idaes.
- **Docs:**
  - explanation/flexpse_integration (the hand-off, time zones, interval rules, scopes);
  - how-to guides: prepare_fit_data, apply_to_a_flexpse_model;
  - intersphinx links into the flex-pse docs.
- **Tutorials:** 07 From raw data to a flex-pse model, in a plain script: clean and calibrate, `to_grid` with a
  facility time zone and a gap rule, `apply_values`, solve with HiGHS; then `fit_data` and a flex-pse regressor,
  scored with `split_holdout` and `metrics`.
- **After M3 (in flex-orchestrator):** it pins flex-signal 0.3.0 and replaces its own copies, following
  `FLEX_ORCHESTRATOR_OVERLAP.md` §5. That work is tracked in flex-orchestrator, not here.

### M4: Forecasting (0.4.0)

**Status (2026-11-16): built.** The four forecast steps, `fs.backtest`, the docs and tutorial 08 are in. The
backtest acceptance test passes on synthetic data, and a forecast is fixed into a flex-pse 0.3.1 model and solved
with HiGHS. The biogas report waits on O1; the `[ml]` and `[nixtla]` adapters on O2.

A forecast is a prediction of a stream's future values. It is written into the frame's future rows, flagged
`FORECAST`, and reaches a flex-pse model the same way measured data does: `handoff.to_grid`, then `apply_values`
fixes the model's variables to it (D24). Only libraries that support pandas 3 are used (D23).

- **Scope:**
  - `flexsignal.steps.forecast` (`fs.forecast`), steps that share these fields (D25):
    - `streams` (the targets), `horizon`, and `issue_time`: the time the forecast is made from. By default it is
      the last time any target has a value. Target values after it are ignored, and the rows from it to
      `issue_time + horizon` are written, extending the index on the data's step if needed.
    - `exog`: streams known ahead, such as a price schedule or a weather forecast. Their values after
      `issue_time` are rows of the input; a missing one is an error naming the stream and the step.
    - `quantiles`, e.g. `[0.1, 0.9]`, written as `<stream>__q10` and `<stream>__q90` columns.
    - The target's metadata records `issue_time`; the forecast rows are flagged `FORECAST`.
  - Methods:
    - `persistence` and `seasonal_naive`: the baselines, in core;
    - `structural`: statsmodels' `UnobservedComponents` (level or trend, seasonal cycles, an AR term, `exog`).
      `fit` estimates its parameters and stores them as JSON; `transform` filters the new history with them;
    - `regression`: a linear model (the `LINEAR_MODELS` keys) of Fourier cycles, `exog`, the target's own lags
      (forecast step by step) and other streams' lags no shorter than the horizon.
  - Intervals: from the model's forecast variance (`structural`), or from the training residuals, widened with
    lead time where the method defines it (baselines). `regression`'s intervals don't widen with lead time; the
    docs say so.
  - `fs.backtest(spec, data, folds=..., every=..., train=..., compare=[...])` in `flexsignal.eval`: blocked,
    time-ordered folds ending at the end of the data, each issued `every` apart (default: the horizon). For each
    fold, every stream except `exog` is hidden after the issue time, the whole pipeline is fitted and run, and
    the forecast is scored against the input data. `compare` swaps other forecast steps (the baselines, say)
    into the same pipeline, so cleaning is shared. The result has a scores table per method and fold, a summary
    with skill against a chosen baseline, the forecasts themselves, and `to_dict()`.
  - `GridValues.estimated` also counts steps resting on forecast values.
  - Not in scope (moved): skforecast (`[ml]`) and statsforecast/mlforecast (`[nixtla]`), which require
    `pandas<3` (O2); SARIMAX (`structural`'s AR term and `exog` cover dynamic regression); `dispatch` elements
    (D24).
- **Acceptance:**
  - On synthetic data with known daily and weekly cycles, `structural` and `regression` beat `seasonal_naive` in a
    backtest. The biogas report waits on O1.
  - A forecast reaches a flex-pse model as fixed values: `to_grid` → `apply_values` → solve with HiGHS.
  - Forecasts and backtests are identical across runs (no randomness, or fixed seeds).
  - `load_spec(dump_spec(s)) == s` and Python/JSON parity for every forecast step; fitted state is JSON.
  - Leakage: a backtest's forecast doesn't change when the data after the issue time changes, except `exog`.
- **Docs:** explanation/forecasting_and_backtesting; how_to/forecast_and_backtest; the forecast steps in the
  reference.
- **Tutorials:** 08 Forecasting and backtesting: the baselines, `structural` and `regression` on a plant with
  daily and weekly cycles, a backtest against seasonal-naive, intervals, and the forecast fixed into a flex-pse
  model.

### M5: Advanced backends (0.5.0)

**Status (2026-11-17): built.** The extension API, `[units]`, and nine adapters registered through it:
`PyodDetector`, `RupturesChangepoint`, `StumpyDiscord` (`[anomaly]`), `PypotsImpute`, `PypotsForecast`
(`[pypots]`), `ChronosForecast` (`[foundation]`), `T0Forecast` (`[t0]`), `NeuralforecastForecast` (`[deep]`), `DartsForecast` and
`SktimeForecast`. Every acceptance item passes except the biogas half of the SAITS benchmark, which waits on O1.
Adapters train on the data they are given and store nothing (D26). Tutorials 09 and 10 run live (about 40 s and
20 s), so no precomputed results are committed.

- **Scope:**
  - `[pypots]`: a windowing adapter (DataFrame ↔ `{"X": [n, T, F]}` with the index restored) and SAITS, BRITS,
    CSDI and TimesNet for imputation and forecasting.
  - `[foundation]`: Chronos-2 by default; `[t0]`: t0-alpha. Open weights only (D29).
  - `[deep]`: neuralforecast.
  - `[anomaly]`: pyod `detector`, ruptures `changepoint`, stumpy.
  - `[darts]` and `[sktime]` adapters.
  - `[units]`: pint conversion at the boundary.
  - The extension API (§3.8, D21): `register_cleaner`, `register_imputer`, `register_forecaster`,
    `register_model`, options generated from the function signature, `fs.load_plugins` / `plugins=`, the
    `requires` hint, provenance records naming the function, and the adapters above registered through it.
- **Acceptance:**
  - Each adapter has a test that is skipped cleanly when its extra is missing, and a CI matrix job installs
    `[all]`.
  - A cleaner, an imputer, a forecaster and a model registered from a test module run from a JSON spec with
    output identical to the same steps built in Python, and save, load and transform as fitted pipelines.
  - Loading a spec that names an unregistered method fails with the "register it first" error; a spec can't make
    flex-signal import a module (contract test, alongside the D15 condition checks).
  - Invalid options to a registered method fail on load with the field named.
  - The PyPOTS windowing round trip preserves the index.
  - The benchmark compares SAITS with the `tiered` imputer on synthetic and biogas data.
- **Docs:** explanation/extending (the "only registered methods run" rule), how_to/add_your_own_method, and
  the generated backend table, with the extra for every backend. The step and backend tables list registered
  methods too, marked as such.
- **Tutorials:**
  - 09 Deep imputation and forecasting with PyPOTS and Chronos, compared with the core methods. If training is
    too slow for CI, the tutorial loads committed, precomputed results, and a test checks that they still
    reproduce.
  - 10 Your own methods: register a cleaner, an imputer and a model; use them from Python and from a JSON spec;
    score them against the injected faults and a built-in method; save and reload the fitted pipeline; see what
    is refused (an unregistered kind, bad options); and package the methods so `pip install` makes them
    available.

### M6: Stable release (1.0.0)

- **Scope:**
  - Freeze `schema_version` at 1.0.0, with migrations from every 0.x schema.
  - Complete the orchestrator contract suite (§4.3), plus an optional test against flex-orchestrator's memory
    backend.
  - Write a deprecation policy.
  - Finish the docs and examples, and make `publish.yml` publish to PyPI through trusted publishing.
- **Acceptance:**
  - Every 0.x example spec loads and migrates.
  - The contract suite is green.
  - `pip install flex-signal` from PyPI works.
  - flex-orchestrator maintainers sign off that the API is enough for their wrapper.
- **Docs:** explanation/orchestrator_contract, a migration guide, and a complete reference.
- **Tutorials:** 11 A cleaning spec as a flex-orchestrator study: the run/fit contract, payloads, and migrating
  an old spec. Revisit browser-run tutorials (§7), now that flex-signal is on PyPI.

### M7: Acquirium (1.1.0)

- **Scope:** re-check §4.2 against acquirium `main`, then build:
  - `AcquiriumSource` (polars ↔ SignalFrame);
  - `AcquiriumSink` (`register_streams`, `insert_timeseries_arrow`, `__flags` companion streams);
  - `PipelineApp`, a stable class configured by parameters;
  - forecast issuances stored as experiment runs.
- **Acceptance:**
  - A `PipelineApp` deployed on a local Acquirium server, fed synthetic data by a test driver written in this
    repo, produces cleaned streams and flag streams.
  - A forecast run is stored and read back.
  - Redeploying with new parameters doesn't trigger a digest mismatch.
- **Docs:** explanation/acquirium_integration and how_to/deploy_a_pipeline_app.
- **Tutorials:** 12 A live Acquirium soft sensor: a `PipelineApp` that cleans a stream and stores a forecast,
  run on the in-repo test driver.

### Test data

flex-signal's tests, examples, benchmarks and docs use exactly two sources of data. Nothing else
from other repositories, and no plant or customer data, is allowed.

1. **flex-pse's public test dataset.** This is `src/flexparameterize/tests/test_time_series_data/` in flex-pse:
   - `bio_gas_generation.csv` (hourly, raw), adapted from the public
     [Biogas-Time-Series-ML-Engine](https://github.com/MuneebX00/Biogas-Time-Series-ML-Engine) repo;
   - `imputed_bio_gas_generation.csv`, its 15-minute interpolation.

   We vendor these into `src/flexsignal/tests/data/flexpse/` in M1 and do not read them from a flex-pse install,
   so tests have no flex-pse dependency. A `README` there records the source, the flex-pse commit they were copied
   from, and their licence. Before vendoring, confirm the upstream repo's licence permits redistribution.
2. **`flexsignal.testing.synthetic`**, a seeded generator built in M1, which produces plant-like multivariate
   series:
   - daily and weekly seasonality and noise;
   - correlated streams and on/off regimes (the C and D streams of §3.5);
   - an intermittent trusted reference;
   - injected faults (spikes, stuck sensors, drift, steps, gaps of controlled length).

Each series comes with its ground truth. Because the truth is known, acceptance tests can assert accuracy (for
example "`A_clean` is within tolerance of the true A"), which real data cannot support. Generated fixtures are
deterministic for a given seed. Small ones may be committed as Parquet under the package's `tests/data/`. Users
can run any pipeline on their own data, but none of it is shipped here.

---

## 9. Decisions and open questions

### Decided (2026-11-06)

| # | Decision |
|---|---|
| D1 | **flex-pse imports flex-signal, never the reverse at import time.** Core has no flex-pse dependency. We mirror the flexcore conventions (exceptions, `_StrictModel`, `SourceRef`, `Quantity`) rather than import them. The only direct contact points are the duck-typed `TimeBlock` grid and handing data to flexparameterize (§4.1). Enforced by import-linter |
| D2 | **Calibrate A to a trusted, intermittent reference B.** `calibrate` fits A→B where B is valid and `fit_when` holds, uses B directly where it is available (`prefer_reference`), and optionally refits on a rolling window to follow drift (§3.5) |
| D3 | **Always track the latest flex-pse release** (0.3.1 as of 2026-11-15, tag `0.3.1`, the same pin as flex-orchestrator). Each new flex-pse release bumps the pin, the tests and the docs in the same PR. A scheduled job tests against flex-pse `main` to catch breakage early. ARIMA-specific helpers wait for a release that ships ARIMA |
| D4 | **Import name `flexsignal`**, matching flex-pse. flex-orchestrator will adopt the same naming |
| D5 | **pandas internally**; polars only at the Acquirium boundary |
| D6 | **No orchestrator plugin in flex-signal.** flex-orchestrator will write a direct wrapper once flex-signal is stable. flex-signal guarantees an orchestrator-ready API (§4.3), verified by contract tests |
| D7 | **Acquirium last.** Build against acquirium `main` in phase 7, after everything else works, which gives its API time to settle. Nothing before phase 7 depends on Acquirium |
| D8 | **Documentation mirrors flex-pse:** Sphinx with furo and MyST, Diátaxis layout, strict `-W` builds, reference tables generated from the code, tested examples, GitHub Pages. Docs ship with every phase (§7) |
| D9 | **Steps are frozen pydantic models; fitted state is a separate JSON dict** (`fit(sf) -> dict`, `transform(sf, state)`). This guarantees Python/JSON parity and keeps specs unchanged by fitting (M1, §3.3) |
| D10 | **Provenance is computed by the pipeline** from each step's input and output, not hand-written by steps (M1, §3.2) |
| D11 | **Tests may use the whole package.** import-linter exempts imports *from* `*.tests.*` modules, and library code stays fully layered. `flexsignal.testing` (the synthetic generator) sits in the `eval \| api \| testing` layer (M1) |
| D12 | **Tutorials are marimo notebooks in this repo, built with each milestone** (2026-11-10). marimo matches flex-pse-examples, keeps notebooks as plain Python that can be linted and tested, and can export static or browser-run pages. Each milestone's tutorials are part of its definition of done (§7, §8) |
| D13 | **Rows added by joining inputs are flagged `NO_SAMPLE`** (2026-11-10), together with `MISSING`, and `align`, `resample` and `interpolate` skip them. Found while building tutorial 04: without the flag, an off-grid source aligned to the join's padding and came out all NaN (§3.2) |
| D14 | **Naive timestamps are taken as the machine's local time, with a warning** (2026-11-10), instead of being refused. flex-pse work usually runs on a "local system" at, or set up for, one plant, and rarely mixes time zones, so refusing was friction without benefit. The zone comes from `tzlocal` (an IANA name, so daylight saving is handled), and `TZ` overrides it. `NaiveTimestampWarning` says which zone was assumed; `timezone=` silences it. Timestamps written in a spec still mean UTC, so a spec means the same on every machine. Daylight-saving gaps and repeats in naive data are still errors |
| D15 | **Conditions are checked against an allowlist before they are evaluated** (2026-11-10). `pandas.eval` let a condition reach Python internals through attribute access (`flow.__class__.__init__.__globals__`) and call methods (`flow.abs()`), so a shared spec could run code. `check_condition` parses each condition with `ast` and accepts only stream names, numbers, arithmetic, comparisons, boolean operators and the functions in `CONDITION_FUNCTIONS`, with exponents of at most 100. It runs when a spec is loaded and again before evaluation. File paths in specs remain a trust issue (docs: spec_schema) |
| D16 | **flex-signal evaluates conditions itself** (M2, 2026-11-11). The parsed, allowlisted tree is evaluated with numpy instead of being handed to `pandas.eval`. This removes the dependency on `pandas.eval`, lets helpers like `held()` read the time index and the flags, and lets conditions be type-checked when a spec loads. `&` and `|` keep `pandas.eval`'s low precedence, and every comparison with a missing value is False, `!=` included |
| D17 | **Spec models are allowlisted keys, and fitted models are coefficients** (M2). `calibrate` and `regression` accept only registered linear-model keys (sklearn, statsmodels) or `scipy:curve_fit` with a checked formula, never a `"pkg.mod:Class"` path, so loading a spec never imports code. Every accepted model is linear in its coefficients, or a formula, so the fitted state is plain JSON numbers and applying it needs only numpy. Non-linear learned models (GBM and others) wait for a safe binary format such as skops |
| D18 | **New flag `FILTERED`** (M2) for values a smoothing filter rewrote, so `is_raw()` and downstream users can tell them from readings. It is bit 14, which leaves one bit (15) free |
| D19 | **`knn` and `iterative` are fitted on the data they fill** (M2). scikit-learn's fitted imputers can't be stored as JSON, so these two learn at transform time and store nothing at fit. `regression` and `kalman` are the fitted alternatives |
| D20 | **The imputer benchmark is built in, without pygrinder** (M2). Hiding block and random gaps takes a few lines. pygrinder's richer missing-data patterns come with `[pypots]` in M5, so there is no `[eval]` extra |
| D21 | **Users add methods by registering them; only registered methods run** (2026-11-15). Custom cleaners, imputers, forecasters and models are registered in code (`fs.register_imputer("CustomMethod", fn)`, an installed package's entry point, or `plugins=` named by the caller) before a spec loads, and a spec refers to them by name. Specs never name import paths, so a spec from someone else can run only what the person running it chose to register. The function's signature defines the step's validated options, and flex-signal does the bookkeeping (gaps, flags, provenance). Replaces the `"pkg.mod:fn"` derive step and the idea of letting specs name classes (§3.8, M5) |
| D22 | **flex-signal owns handing data to models; flex-orchestrator depends on flex-signal** (2026-11-15). flex-orchestrator already lined stored data up with model time steps, built `fix` elements and prepared fit data, which overlapped M3's plan. Everything must also work in a plain script, so those data functions move to `flexsignal.handoff` and `flexsignal.integrations.flexpse` with their tests and behaviour unchanged, and flex-orchestrator calls them. flex-signal stops at a prepared `X, y`: regressor wrappers stay in flex-orchestrator for sweeps and automated fits. Design-spec grids, store selectors, studies, sweeps and Ray stay in flex-orchestrator. Inventory and migration plan: `FLEX_ORCHESTRATOR_OVERLAP.md`. Replaces M3's `to_grid` / `to_parameterize` / `to_elements` / `patch_spec` plan |
| D23 | **Only libraries that support pandas 3** (M4, 2026-11-16). flex-signal and flex-pse both require `pandas>=3`. skforecast 0.27, statsforecast 2.1 and mlforecast 1.1 require `pandas<3`, so installing them would downgrade pandas under flex-pse. Their extras (`[ml]`, `[nixtla]`) wait until they support pandas 3 (O2). M4's forecasters use statsmodels, scikit-learn and scipy, which are core dependencies already |
| D24 | **A forecast reaches a model as fixed values, not `dispatch`** (M4, 2026-11-16). A forecast is a prediction, so it fixes model variables like measured data does, through `to_grid` and `apply_values`. flex-pse's `dispatch` element is for setpoints an external controller sends (and has no units), which is not flex-signal's job |
| D25 | **A forecast step writes the frame's future rows** (M4, 2026-11-16). Forecasting is a step like any other, so it runs from a JSON spec after cleaning. Values known ahead (`exog`) are simply rows after the issue time in the input. A backtest hides everything except `exog` after each issue time, so a forecast can't see the future it is scored on |
| D26 | **Adapters to learned models train on the data they are given and store nothing** (M5, 2026-11-17). Trained weights and fitted library objects (PyPOTS and neuralforecast networks, pyod detectors, darts and sktime models) can't be JSON, and D9 keeps fitted state JSON. So, as with `knn` and `iterative` (D19), they are trained inside `transform`, with fixed seeds on CPU, and a fitted pipeline stores nothing for them. A backtest refits every fold anyway. Pretrained foundation models are named by Hugging Face id and a pinned revision, so they are reproducible without stored state. A safe binary format for weights can come later |
| D27 | **A registered method's name is its kind; flex-signal's adapters prefix theirs with the library** (M5, 2026-11-17). One name, e.g. `PyodDetector`, is the JSON `kind` and `fs.clean.PyodDetector`, so registered methods read differently from built-in kinds (`hampel`) and are marked "registered" in the step table. A name can't shadow a built-in step; registering the same function again (a re-run notebook cell) replaces it. The JSON Schema lists flex-signal's own steps, adapters included, but not users' |
| D28 | **Units are converted only where asked** (M5, 2026-11-17). Values stay plain float64 with units as metadata (§3.2). pint is imported only by a `convert_units` step or a `range` bound in other units; offset units (degC, K) convert correctly. flex-pse converts the values it is given itself, so the hand-off doesn't convert |
| D29 | **Adapters only to models with open weights** (M5, 2026-11-18). flex-signal ships adapters only to models whose weights are free for commercial use (Apache-2.0: Chronos-2, TimesFM 2.5, t0-alpha). Models under non-commercial or revenue-capped licences (TimesFM-3, Mistral's Timestral) get no adapter and no licence switch: users who may use them register their own forecaster (§3.8). This replaces the TimesFM-3 `accept_licence` guard |
| D30 | **Adapters don't block a library's untested models** (M5, 2026-11-18). An adapter's model option takes any model class of its library by bare name: the tested ones (listed in the docs) run with flex-signal's simple settings, any other with its own settings in `params`, and a missing required setting is named in the error. The name is looked up only among that library's model classes (a subclass of its base model), and the bare-identifier pattern keeps import paths out of specs (D21). Untested models aren't in CI; licences of weights they download are the user's to check (D29) |

### Open

| # | Question |
|---|---|
| O1 | **The biogas test data** (2026-11-11). The plan assumed the data ships in a flex-pse release. It doesn't yet: `bio_gas_generation.csv` and `imputed_bio_gas_generation.csv` arrived on flex-pse `main` in "Arima integration (#102)" on 2026-11-10, after the latest release tag. Their upstream source, MuneebX00/Biogas-Time-Series-ML-Engine, still needs its licence confirmed before we vendor them. Until both are settled, M2's two biogas acceptance items and tutorial 06's biogas section are pending; everything else in M2 runs on synthetic data |
| O2 | **Forecasting libraries that require pandas 2** (2026-11-16). skforecast, statsforecast and mlforecast require `pandas<3` (D23). Revisit when they support pandas 3, then add the `[ml]` and `[nixtla]` adapters. Until then, the extension API (§3.8) lets a user register a forecaster of their own |
| O3 | **Small gaps left in the extension API** (M5, 2026-11-17). (a) A registered forecaster isn't told which quantiles were requested, so the PyPOTS, darts and sktime adapters refuse unavailable quantiles through the order in which flex-signal reads their result; passing `quantiles` to functions that declare it would be cleaner. (b) Library `params` dicts are checked when the step runs, not when the spec loads, because checking them needs the library. (c) The TimesFM 2.5 run path is written but untested: `timesfm` isn't installed in CI. (d) pygrinder's missing-data patterns aren't in `benchmark_imputers` yet |
