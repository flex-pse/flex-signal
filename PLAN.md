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
  - cleaned or forecast data → flex-pse `fix`/`dispatch` elements, spec patches and ARIMA history;
  - cleaned data → `flexparameterize.apply_to_model` and surrogate fits;
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
| flex-pse `flexops` | `apply_spec(model, elements)` applies `fix`, `unfix` and `dispatch` to a live model. `set_external_dispatch(var, series)`. Price and exogenous profiles are construction options, set before build. `TimeBlock` uses a naive local-time index, so `index_of` raises on off-grid timestamps | Our flex-pse adapter emits element dicts aligned to `TimeBlock.datetime_index`, or patches the spec before build. |
| flex-pse `flexparameterize` (v0.2.0, our target) | `TagMap`, `check_sufficiency`, `apply_to_model(model, df, tagmap)`, a `Regressor` protocol, and the `constant_intensity` and `multilinear` regressors. ARIMA is reserved in v0.2.0. On flex-pse's `arima_integration` branch, `ArimaRegressor` **refuses NaNs and uneven steps** and tells the user to "resample and impute" | flex-signal is that missing "resample and impute" step. Our output must be a regular-grid DataFrame with no gaps and columns named by model alias. |
| flex-pse `test_time_series_data/impute_timeseries.py` | Standalone linear-interpolation script | Replace it with a flex-signal pipeline. |
| flex-orchestrator | Plugin system: entry-point group `flex_orchestrator.plugins`, `@command(params=PydanticModel)`, `orch.store_data` / `get_data`, events. **The `StudyConfig` sweep union is closed.** `fitting/prep.py` and `inputs/resolve.py` already do basic align, resample and gap filling | flex-orchestrator will wrap flex-signal directly once flex-signal is stable. We ship no plugin; instead our API follows the orchestrator's calling conventions (§4.3). |
| Acquirium (alpha, 0.4.0a8; build against `upstream/main`) | Python client returns **polars**. Data is stored in UTC. Writes need `register_streams` and then `insert_timeseries_arrow`. Derived streams are `aq.App` subclasses running **inside the server**; their module digest is pinned, and outputs must contain no nulls. There are no quality-flag columns and only one value per `(stream, ts)` | Use pandas internally and convert at the boundary. Ship one stable `PipelineApp` class whose behaviour comes entirely from parameters (the spec). Write flags as separate streams. Store forecast issuances as experiment runs or as per-horizon streams. |
| flows-prep (legacy, py<3.12) | Tukey and rate-of-change outlier removal, stuck-value detection, short-gap interpolation with a consecutive-NaN limit, lagged ElasticNet imputation, Fourier features, blocked CV, clipping to bounds. PyPOTS was planned but never committed | This is the method inventory to port as thin steps. Keep its config ideas, not its code. |
| flows-mpc (legacy) | statsforecast MSTL, statsmodels, a Keras seq2seq model, MLflow tracking | Confirms Nixtla and statsmodels as the forecasting backbone. Drop TensorFlow and use torch-based extras. |
| svcw-app | Driver for the SVCW SCADA CSV (38 hourly tags, 2020–2023, with gaps and duplicates) | Real-data integration fixture. |

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
  eval/            masking benchmark for imputers (pygrinder), forecast backtesting metrics
  integrations/
    flexpse/       elements, spec patching, ARIMA history, flexparameterize hand-off
    acquirium/     source, sink, PipelineApp (last phase)
  api.py           run / fit / backtest entry points: the orchestrator-ready contract (§4.3)
```

import-linter layering (same tool as flex-pse):

```
core <- spec <- backends <- steps <- pipeline <- {io, eval, api} <- integrations.{flexpse | acquirium}
```

Each `integrations.*` package imports its third-party target only inside itself. Each is behind its own extra.

**Import direction with flex-pse.** flex-pse workflows import flex-signal, so flex-signal must never import flex-pse
at module import time. This rules out circular imports in either direction. import-linter enforces two rules:

- no module outside `integrations.flexpse` may import `flexcore`, `flexops` or `flexparameterize`;
- `integrations.flexpse` itself imports flex-pse only lazily, inside functions, and only for `flexparameterize`.
  The time grid is duck-typed (§4.1).

`import flexsignal` therefore never pulls in pyomo or idaes.

### 3.2 Data model: `SignalFrame`

We do **not** create a new timeseries library. A `SignalFrame` is a thin wrapper around plain pandas objects:

- `values`: `pd.DataFrame`, tz-aware `DatetimeIndex` (UTC internally), one float column per stream.
- `flags`: `pd.DataFrame` of the same shape, `uint16` bitmask per sample.
- `meta`: `dict[str, StreamMeta]` holding units (pyomo/pint-parseable string, e.g. `"m^3/hr"`), quantity kind, source
  reference (Acquirium `ref_uri` or a file), and a description.
- `history`: an ordered list of the step records that touched each column (step name, spec hash, parameters).
  This is the provenance.

Quality flags are an `IntFlag` bitmask:

```
RAW=0  MISSING  OUT_OF_RANGE  RATE_OF_CHANGE  OUTLIER  STUCK  MANUAL
CORRECTED  IMPUTED_SHORT  IMPUTED_MODEL  FORECAST  DERIVED  RESAMPLED
```

A cleaning step does not delete data. It sets the value to NaN and sets a flag bit. An imputer fills NaNs and
sets `IMPUTED_*`. A downstream consumer can therefore always ask which values are real.

Steps receive and return a `SignalFrame`. `sf.df` / `sf.to_pandas()` give the bare DataFrame for any library
call. The Acquirium adapter converts polars to and from this frame at the boundary.

### 3.3 Step protocol

Steps follow the scikit-learn contract, so users can drop to raw sklearn when they need to:

```python
class Step(Protocol):
    name: str
    def fit(self, sf: SignalFrame) -> "Step": ...          # stateless steps: no-op
    def transform(self, sf: SignalFrame) -> SignalFrame: ...
    def to_spec(self) -> StepSpec: ...                      # declarative config
    def fitted_state(self) -> dict | Artifact: ...          # JSON params where possible, else native save

class Forecaster(Step, Protocol):
    def predict(self, sf: SignalFrame, horizon: str | int, *, exog_future: SignalFrame | None = None) -> SignalFrame: ...
```

- **Backends** are registered by a string key, for example `"sklearn:HuberRegressor"`, `"statsmodels:SARIMAX"`,
  `"pypots:SAITS"`, `"nixtla:AutoETS"`, `"chronos:chronos-2"`, or `"pkg.mod:Class"` for user code. Each key maps
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

Conditions use a small expression language that compiles to a boolean mask. It uses
`pandas.DataFrame.eval`/`query` syntax over stream names: `"C == 0 and D == 0 and B > 0"`. On top of that it adds:

- duration helpers: `held("C == 0", "30min")` means the condition has been true for at least 30 minutes;
- flag predicates: `is_raw("B")`, `not flagged("A", "OUTLIER")`.

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
   for example with statsmodels `RecursiveLS` or a windowed sklearn fit. The calibration then follows drift
   between reference periods.

`report()` covers how many samples were fit, residual statistics, and for each sample whether it was referenced,
calibrated or still missing.

```python
import flexsignal as fs

pipe = fs.Pipeline(name="stream_a_cleanup", steps=[
    fs.select.Align(freq="1min", how="inner"),                    # match B to A by time index
    fs.clean.Range("B", min=0),                                    # basic hygiene first
    fs.clean.Hampel(["A", "B"], window="15min", n_sigma=3),
    fs.correct.Calibrate(
        name="A_clean",
        target="A",                         # stream with the variable error
        reference="B",                      # trusted signal while the condition holds
        features=["A"],                     # model input(s); add others if the error depends on them
        fit_when="C == 0 and D == 0 and B > 0",
        apply_when=None,                    # None = apply everywhere A is valid
        model="sklearn:HuberRegressor",     # or "scipy:curve_fit:mypkg.forms:temp_comp", "statsmodels:RLM", ...
        prefer_reference=True,              # use B itself where B is valid; calibrated A elsewhere
        refit="static",                     # or {"rolling": "30D"} to follow drift between reference periods
        output="new",                       # write A_clean as a new stream (default) or "replace" A
    ),
    fs.impute.Tiered("A_clean", short=("interpolate", "15min"), long="kalman"),
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
    {"kind": "calibrate", "name": "A_clean", "target": "A", "reference": "B", "features": ["A"],
     "fit_when": "C == 0 and D == 0 and B > 0", "model": "sklearn:HuberRegressor",
     "prefer_reference": true, "refit": "static"},
    {"kind": "impute_tiered", "streams": ["A_clean"], "short": {"method": "interpolate", "max_gap": "15min"},
     "long": {"method": "kalman"}}
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

- `Forecast` step: `target`, `exog` (past and known-future covariates), `horizon`, `freq`, `backend`, `params`,
  `quantiles` (optional), and `issue_time` semantics.
- `Backtest(forecaster, folds=..., horizon=...)` → metrics table. It uses blocked, time-ordered folds, the same idea
  as flows-prep's `get_train_test_folds`. Its output feeds flex-pse model selection.
- The output is a `SignalFrame` flagged `FORECAST` that carries `issue_time` metadata. Optional quantile columns
  are named `<stream>__q10` and so on.

---

## 4. Ecosystem integration

### 4.1 flex-pse (`flex-signal[flexpse]`, targeting flex-pse v0.2.0)

flex-signal has two direct points of contact with flex-pse:

1. **Aligning to the time grid.** flex-signal resamples data onto a flex-pse `TimeBlock`. `TimeBlock` is an IDAES
   block, and importing it pulls in pyomo and idaes, so flex-signal never imports it. Instead, `to_grid(sf, grid)`
   duck-types `grid` and accepts any of:
   - an object with a `.datetime_index` attribute (`TimeBlockData`, or a model via `find_time_block`, called by the
     flex-pse side);
   - a `pd.DatetimeIndex`;
   - a flex-pse `time` element dict `{start_date, end_date, time_step}`.
2. **Handing data to flexparameterize.** `to_parameterize(sf, tagmap=None)` returns the gap-free DataFrame that
   `flexparameterize.apply_to_model` and the regressors expect: a DatetimeIndex with model-alias columns. If a
   `TagMap` is given, the column mapping is delegated to it. This is the only place flexparameterize is imported,
   and only lazily.

Everything else is plain data with no flex-pse import, and the flex-pse workflow applies it:

```python
from flexsignal.integrations import flexpse as fsp

grid = fsp.to_grid(clean, model.fs.time)             # resample + check: no NaN, exact grid
data = fsp.to_parameterize(clean, tagmap)            # -> flexparameterize.apply_to_model(model, data, tagmap)

# Plain-dict spec elements; the flex-pse caller passes them to flexops.apply_spec or adds them to a FlowsheetSpec
elements = fsp.to_elements(grid, mapping={
    "A_clean": {"kind": "fix", "variable": "plant.feed.flow_vol", "units": "m^3/hr"},
    "price":   {"kind": "dispatch", "unit": "grid", "variable": "electricity"},
})

# Before build: return a patched copy of a FlowsheetSpec *dict* (energy_prices, capacity_factor, Feed limits)
spec_dict = fsp.patch_spec(spec_dict, grid, {"costing.energy_prices.electrical": "price"})
```

`arima_history(...)`, for seeding an ARIMA surrogate, is deferred until flex-pse ships the ARIMA surrogate past
v0.2.0.

Rules the adapter enforces:

- Convert UTC to the facility's local time and make the index naive, because the orchestrator treats flex-pse
  time as local wall clock.
- Resample onto `TimeBlock.datetime_index`.
- Refuse leftover NaNs with a `FlexDataError` that names the gap and suggests an imputer.
- Pass units through as pyomo strings (`^`, not `**`).

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

flex-signal ships **no orchestrator code**. Once flex-signal has a stable release, flex-orchestrator will add a
direct wrapper on its own side, as commands and/or a `signals:` stage in `StudyConfig`. Our job is to make that
wrapper trivial. The public API therefore follows this contract, which matches how the orchestrator already calls
flex-pse and its regressors:

| Contract | What it means for our API |
|---|---|
| **Pure functions with a spec in and JSON-able results out** | `flexsignal.run(spec, data) -> RunResult`, `flexsignal.fit(spec, data) -> FittedPipeline`, `fitted.transform(data)`, `fitted.forecast(data, issue_time=, horizon=)`, `flexsignal.backtest(spec, data, folds=)`. `spec` may be a `SignalSpec`, a dict or a path. These map one-to-one onto orchestrator `@command`s whose params model simply embeds `SignalSpec` |
| **No I/O inside the pipeline when data is passed in** | Spec `inputs` are *named slots*, e.g. `{"kind": "frame", "name": "historian"}`. The caller binds them: `data={"historian": (df, units)}`. Orchestrator `Selector`s resolve to these slots. File and Acquirium inputs exist only for standalone use |
| **The orchestrator's data shape** | Data goes in and out as a `pd.DataFrame` with a tz-aware UTC `DatetimeIndex` plus `units: dict[str, str]`, exactly what `orch.get_data` / `orch.store_data(name, df, units=)` use. `RunResult` exposes `.data`, `.units`, `.flags` (DataFrame), `.report` (JSON-able dict), `.spec_hash`, `.provenance` |
| **Storage-agnostic fitted state** | `FittedPipeline.to_payload() -> {"spec": dict, "state": dict, "blobs": dict[str, bytes]}` and `FittedPipeline.from_payload(...)`. Directory save/load is built on top of this. The orchestrator can then store fits in any backend (file, memory, Acquirium) |
| **Ray-safe** | No global mutable state. Fitted pipelines are picklable. Random seeds are explicit in the spec. Backends are imported lazily on workers |
| **Progress and errors** | An optional `progress: Callable[[float, str], None]` that the orchestrator's `Progress` adapts to. All errors subclass `SignalError` with `field=` dotted paths, so the command ledger records something actionable |
| **Embeddable schema** | `SignalSpec` and its JSON Schema are exported with stable `kind` names and `schema_version` migrations, so `StudyConfig` can embed a `signals:` list without redefining it |
| **Time semantics** | UTC internally. Local naive time appears only at `flexpse.to_grid(..., timezone=)`, matching the orchestrator's `model_inputs.timezone` handling |

**Verification:** `tests/contract/` exercises this contract without importing flex-orchestrator. It wraps the
API in a pydantic params model, JSON round-trips specs, results and payloads, and pickles a fitted pipeline. An
optional integration test runs against flex-orchestrator's memory backend when it is installed.

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
| `pypots` | pypots (+ pygrinder) | Deep imputation and forecasting (SAITS, BRITS, CSDI, TimesNet). Pulls in torch and transformers |
| `eval` | pygrinder | Masking-based imputer benchmarking (MCAR and block gaps) without torch |
| `ml` | skforecast, lightgbm | Lightweight, pandas-native ML forecasting with exogenous variables |
| `nixtla` | statsforecast, mlforecast | Fast AutoARIMA, ETS, MSTL and Theta. Successor to the flows-mpc stack |
| `deep` | neuralforecast | NHITS, TiDE, PatchTST |
| `foundation` | chronos-forecasting (Chronos-2); timesfm optional | Zero-shot forecasting. **TimesFM-3 weights are non-commercial**; we guard against them and default to Chronos (Apache-2.0) |
| `darts` | darts | Wide model zoo through an adapter. Heavy, and opt-in only |
| `sktime` | sktime | Adapter to the sktime estimator zoo |
| `anomaly` | pyod, ruptures, stumpy | ML outlier detectors, change points (regime and drift), matrix-profile discords |
| `features` | tsfresh | Feature extraction for regression imputers |
| `units` | pint, pint-pandas | Boundary unit conversion. Pint 0.26 needs Python ≥3.12 |
| `symbolic` | pysr | Symbolic-regression correction models. Needs a Julia runtime |
| `flexpse` | `flex-pse[parameterize]==0.2.0` | Used only in tests and for `to_parameterize`. flex-pse is the importer, not a dependency of core. The pin is bumped as flex-pse releases, after its integration tests pass |
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
| Clean | `filter` (savgol, butterworth `sosfiltfilt`, median) | scipy.signal |
| Clean | `changepoint` | ruptures `[anomaly]` |
| Clean | `detector` | pyod `[anomaly]` |
| Clean | `detotalize` (totalizer → rate) | pandas (from flows-prep) |
| Correct | `calibrate` (map/residual, `fit_when`/`apply_when`) | sklearn / statsmodels / `scipy.optimize.curve_fit` / pysr |
| Correct | `mass_balance` (reconcile children to parent) | scipy.optimize least squares |
| Impute | `interpolate` (linear/time/pchip/akima, `max_gap`) | pandas / scipy |
| Impute | `kalman` | statsmodels state space (`UnobservedComponents` / `SARIMAX` smoother) |
| Impute | `regression` (lags, Fourier, other streams) | sklearn (ElasticNet / GBM) |
| Impute | `knn`, `iterative` | sklearn |
| Impute | `pypots` | PyPOTS `[pypots]` |
| Impute | `tiered` | Composition of the methods above, routed by gap length |
| Forecast | `forecast`, `backtest` | statsmodels, skforecast, statsforecast, neuralforecast, chronos, pypots, darts/sktime adapters |
| Derive | `expression`, `function` (`"pkg.mod:fn"`) | pandas eval / user code |

---

## 7. Roadmap

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0 | Scaffold: pyproject (setuptools, src layout), ruff, black, import-linter, pytest markers (`unit`, `component`, `integration`), CI on 3.11–3.14, CONTRIBUTING and CHANGELOG mirroring flex-pse | `pytest` and the lint hooks pass on an empty package |
| 1 | `core` (SignalFrame, flags, exceptions, registries), `spec` (schema, io, migrations, JSON Schema, named input slots), `select` and `clean` steps, `interpolate`, parquet io, `Pipeline` fit/transform plus `to_payload`/`from_payload`, `api.run`/`api.fit` | The A/B/C/D example runs from Python and from JSON with an identical result; spec round trip passes; JSON Schema is committed; first `tests/contract/` tests pass (JSON round trip, pickling) |
| 2 | `calibrate`, `regression`/`kalman`/`tiered` imputers, `eval` masking benchmark | Imputer comparison report on the SVCW fixture; replaces flex-pse `impute_timeseries.py` |
| 3 | flex-pse v0.2.0 integration: `to_grid`, `to_parameterize`, `to_elements`, `patch_spec` | End-to-end: raw CSV → clean → `flexparameterize.apply_to_model` (multilinear) → `flexops.apply_spec` with our elements. import-linter shows no flex-pse import outside `integrations.flexpse`, and `import flexsignal` loads no pyomo. `arima_history` is added when flex-pse releases ARIMA |
| 4 | Forecasting: statsmodels, skforecast and statsforecast backends, plus `backtest` and `api.backtest` | Backtest table on the SVCW data; forecast applied as `dispatch` to a model |
| 5 | `[pypots]`, `[foundation]`, `[anomaly]`, `[darts]`/`[sktime]` adapters | Each extra has an adapter test, skipped when the extra isn't installed |
| 6 | Hardening for a stable release: complete the orchestrator contract suite (§4.3), docs, the first stable `schema_version`, and an optional test against the orchestrator memory backend | Contract suite green. Hand-off to flex-orchestrator, which writes its own wrapper |
| 7 | Acquirium (against `main`): source, sink and `PipelineApp`; forecast runs stored as experiments | Live App deployed on a local Acquirium server with svcw-app data; flags streams visible |

Test data:

- the SVCW SCADA CSV (from svcw-app);
- the flex-pse `test_time_series_data` series;
- synthetic series with known injected faults (drift, stuck sensor, spikes, gaps), so every cleaner has a
  deterministic unit test.

---

## 8. Decisions and open questions

### Decided (2026-11-06)

| # | Decision |
|---|---|
| D1 | **flex-pse imports flex-signal, never the reverse at import time.** Core has no flex-pse dependency. We mirror the flexcore conventions (exceptions, `_StrictModel`, `SourceRef`, `Quantity`) rather than import them. The only direct contact points are the duck-typed `TimeBlock` grid and handing data to flexparameterize (§4.1). Enforced by import-linter |
| D2 | **Calibrate A to a trusted, intermittent reference B.** `calibrate` fits A→B where B is valid and `fit_when` holds, uses B directly where it is available (`prefer_reference`), and optionally refits on a rolling window to follow drift (§3.5) |
| D3 | **Target flex-pse v0.2.0**, the same pin as flex-orchestrator. CI pins it, and we bump the pin and the tests as flex-pse releases. ARIMA-specific helpers wait for an ARIMA release |
| D4 | **Import name `flexsignal`**, matching flex-pse. flex-orchestrator will adopt the same naming |
| D5 | **pandas internally**; polars only at the Acquirium boundary |
| D6 | **No orchestrator plugin in flex-signal.** flex-orchestrator will write a direct wrapper once flex-signal is stable. flex-signal guarantees an orchestrator-ready API (§4.3), verified by contract tests |
| D7 | **Acquirium last.** Build against acquirium `main` in phase 7, after everything else works, which gives its API time to settle. Nothing before phase 7 depends on Acquirium |

### Open

None at the moment. New questions will be added here as they come up during implementation.
