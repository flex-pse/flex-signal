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
| flex-pse `flexparameterize` (v0.3.0, the latest release) | `TagMap`, `check_sufficiency`, `apply_to_model(model, df, tagmap)`, a `Regressor` protocol, and the `constant_intensity` and `multilinear` regressors. ARIMA is still reserved in v0.3.0. On flex-pse's `arima_integration` branch, `ArimaRegressor` **refuses NaNs and uneven steps** and tells the user to "resample and impute" | flex-signal is that missing "resample and impute" step. Our output must be a regular-grid DataFrame with no gaps and columns named by model alias. |
| flex-pse `flexparameterize/tests/test_time_series_data/` (on the `arima_integration` branch; not in any release yet) | Public hourly biogas data (`bio_gas_generation.csv`: feed volume, TS %, biogas m³/h; ~1,440 rows), its 15-minute linear interpolation, and the `impute_timeseries.py` script that made it | The **one permitted outside dataset** (see §8, "Test data"). flex-signal's `interpolate` step supersedes the script, and must reproduce its output. |
| flex-orchestrator | Plugin system: entry-point group `flex_orchestrator.plugins`, `@command(params=PydanticModel)`, `orch.store_data` / `get_data`, events. **The `StudyConfig` sweep union is closed.** `fitting/prep.py` and `inputs/resolve.py` already do basic align, resample and gap filling | flex-orchestrator will wrap flex-signal directly once flex-signal is stable. We ship no plugin; instead our API follows the orchestrator's calling conventions (§4.3). |
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
  eval/            masking benchmark for imputers (pygrinder), forecast backtesting metrics
  integrations/
    flexpse/       elements, spec patching, ARIMA history, flexparameterize hand-off
    acquirium/     source, sink, PipelineApp (last phase)
  api.py           run / fit / backtest entry points: the orchestrator-ready contract (§4.3)
```

import-linter layering (same tool as flex-pse):

```
core <- {spec | io} <- backends <- steps <- pipeline <- {eval | api} <- integrations.{flexpse | acquirium}
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

### 4.1 flex-pse (always the latest flex-pse release; currently v0.3.0)

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

`arima_history(...)`, for seeding an ARIMA surrogate, is deferred until a flex-pse release ships the ARIMA surrogate.
v0.3.0 does not.

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
| `flexpse` | flex-pse with `parameterize`, **the latest release** (now v0.3.0) | Used only in tests and for `to_parameterize`. flex-pse is the importer, not a dependency of core. flex-pse is not on PyPI, and PyPI rejects git URLs in a published package's metadata. So until flex-pse is published, the pin lives in `environment.yml` and CI as `flex-pse[parameterize] @ git+https://github.com/flex-pse/flex-pse.git@<tag>`, the same way flex-orchestrator pins it. The tag's dist-info still says 0.1.0, so pin by tag, never by `==version`. Once flex-pse is on PyPI, the extra becomes `flex-pse[parameterize]>=<latest>` |
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

## 7. Documentation

The documentation mirrors flex-pse's setup (`docs/`, `.github/workflows/docs.yml`; first mirrored from v0.2.0 and kept in step with each flex-pse release, now v0.3.0), so the two projects
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
                     forecast_and_backtest, feed_flexparameterize, apply_to_a_flexpse_model,
                     save_and_load_a_fitted_pipeline, write_a_backend, deploy_a_pipeline_app (M7)
  examples/          runnable scripts on flexsignal.testing.synthetic data and the vendored flex-pse biogas data
  reference/         flexsignal.core, spec, steps (generated tables), backends, pipeline, api,
                     integrations.flexpse, glossary
  changelog.md       includes CHANGELOG.md
  _ext/ _static/ _templates/autosummary/
```

**Examples are tested.** Every script in `docs/examples/` and every how-to code block runs in CI under the
`integration` marker. Examples that need an optional extra are skipped when it isn't installed.

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
| M3 | 0.3.0 | flex-pse bridge | Synthetic raw data → clean → `flexparameterize` fit → `apply_spec` on a model built with the latest flex-pse release (v0.3.0 today) | M1 (M2 for the full demo) |
| M4 | 0.4.0 | Forecasting | Backtest on synthetic data with known seasonality beats a seasonal-naive baseline; the forecast reaches a model as `dispatch` | M2, M3 |
| M5 | 0.5.0 | Advanced backends | PyPOTS, Chronos, anomaly and darts/sktime adapters pass their tests, skipped when an extra is missing | M2, M4 |
| M6 | 1.0.0 | Stable release | Orchestrator contract suite green; schema frozen at 1.0.0; published to PyPI; hand-off | M1–M5 |
| M7 | 1.1.0 | Acquirium | A live `PipelineApp` runs on a local Acquirium server (`main`), fed synthetic data by an in-repo test driver | M6 |

Every milestone also needs, per the definition of done in §7:

- its docs pages, with `-W` passing;
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

### M2: Calibrate and impute (0.2.0)

- **Scope:**
  - The condition language: pandas `eval` syntax plus `held(...)`, `is_raw(...)` and `flagged(...)`.
  - `calibrate` with `fit_when`/`apply_when`, `prefer_reference`, and `refit` static or rolling. Backends:
    sklearn, statsmodels and `curve_fit`.
  - Imputers: `kalman` (statsmodels), `regression` (lags, Fourier terms, other streams), `knn`, `iterative` and
    `tiered`.
  - `filter` (savgol, butterworth, median) and `detotalize`.
  - `eval`: a built-in masking benchmark (random and block gaps), with pygrinder optional under `[eval]`.
- **Acceptance:**
  - On a synthetic A/B/C/D fixture with known true A and intermittent B, `A_clean` RMSE is under a set threshold.
  - With `refit="rolling"`, the calibration tracks an injected drift that `static` misses.
  - The imputer benchmark report runs on synthetic data and on the flex-pse biogas data.
  - An `interpolate` pipeline reproduces flex-pse's `imputed_bio_gas_generation.csv`.
- **Docs:**
  - getting_started/ten_minutes, built around the A/B/C/D example;
  - explanation pages: conditions, calibration, imputation_tiers;
  - how-to guides: calibrate_against_a_reference, impute_gaps, evaluate_imputers.

### M3: flex-pse bridge (0.3.0)

- **Scope:**
  - `integrations.flexpse`: `to_grid` (duck-typed `TimeBlock`, `DatetimeIndex` or `time` dict, timezone
    handling), `to_parameterize`, `to_elements` (`fix`/`dispatch` dicts) and `patch_spec`.
  - CI job pinned to the latest flex-pse release by git tag (`flex-pse[parameterize] @ git+...@v0.3.0` today;
    see D3), plus a scheduled job against flex-pse `main`.
- **Acceptance:**
  - A component test runs synthetic raw data → clean → `to_grid` → `flexparameterize.apply_to_model` (multilinear) →
    `flexops.apply_spec(model, to_elements(...))`, then solves with HiGHS.
  - An off-grid or gappy input raises `FlexDataError`-style errors that name the gap.
- **Docs:**
  - explanation/flexpse_integration;
  - how-to guides: feed_flexparameterize, apply_to_a_flexpse_model;
  - intersphinx links into the flex-pse docs.

### M4: Forecasting (0.4.0)

- **Scope:**
  - The `forecast` step with past and future exogenous inputs, `issue_time`, quantiles and the `FORECAST` flag.
  - Backends: statsmodels (core), skforecast (`[ml]`) and statsforecast/mlforecast (`[nixtla]`).
  - `backtest` with blocked, time-ordered folds; `api.backtest`.
  - Seasonal-naive and persistence baselines (core).
- **Acceptance:**
  - On synthetic data with known seasonality, backtest metrics beat seasonal-naive for at least one backend. They
    are also reported on the biogas data.
  - A forecast becomes `dispatch` elements on a flex-pse model's grid.
  - Forecasts are deterministic with fixed seeds.
- **Docs:** explanation/forecasting_and_backtesting and how_to/forecast_and_backtest.

### M5: Advanced backends (0.5.0)

- **Scope:**
  - `[pypots]`: a windowing adapter (DataFrame ↔ `{"X": [n, T, F]}` with the index restored) and SAITS, BRITS,
    CSDI and TimesNet for imputation and forecasting.
  - `[foundation]`: Chronos-2 by default, with a TimesFM-3 licence guard.
  - `[deep]`: neuralforecast.
  - `[anomaly]`: pyod `detector`, ruptures `changepoint`, stumpy.
  - `[darts]` and `[sktime]` adapters.
  - `[units]`: pint conversion at the boundary.
- **Acceptance:**
  - Each adapter has a test that is skipped cleanly when its extra is missing, and a CI matrix job installs
    `[all]`.
  - The PyPOTS windowing round trip preserves the index.
  - The benchmark compares SAITS with the `tiered` imputer on synthetic and biogas data.
- **Docs:** how_to/write_a_backend and the generated backend table, with the extra for every backend.

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
| D3 | **Always track the latest flex-pse release** (v0.3.0 as of 2026-11-09, the same pin as flex-orchestrator). Each new flex-pse release bumps the pin, the tests and the docs in the same PR. A scheduled job tests against flex-pse `main` to catch breakage early. ARIMA-specific helpers wait for a release that ships ARIMA |
| D4 | **Import name `flexsignal`**, matching flex-pse. flex-orchestrator will adopt the same naming |
| D5 | **pandas internally**; polars only at the Acquirium boundary |
| D6 | **No orchestrator plugin in flex-signal.** flex-orchestrator will write a direct wrapper once flex-signal is stable. flex-signal guarantees an orchestrator-ready API (§4.3), verified by contract tests |
| D7 | **Acquirium last.** Build against acquirium `main` in phase 7, after everything else works, which gives its API time to settle. Nothing before phase 7 depends on Acquirium |
| D8 | **Documentation mirrors flex-pse:** Sphinx with furo and MyST, Diátaxis layout, strict `-W` builds, reference tables generated from the code, tested examples, GitHub Pages. Docs ship with every phase (§7) |

### Open

None at the moment. New questions will be added here as they come up during implementation.
