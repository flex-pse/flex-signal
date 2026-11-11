# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Core data model (M1).**
  - `SignalFrame`: values, a per-sample `uint16` quality-flag bitmask, per-stream `StreamMeta` (units, quantity
    kind, source) and provenance `history`. The time index is always tz-aware UTC, nanosecond resolution, sorted
    and unique, and frames are immutable.
  - Timestamps without a time zone are taken as this machine's local time (`local_timezone()`, overridable with
    the `TZ` environment variable) and raise a `NaiveTimestampWarning`. Passing `timezone=` silences it. This
    applies to `SignalFrame`, `from_frame`, `read_csv`, `read_parquet` and spec file inputs; timestamps written
    in a spec still mean UTC. New core dependency: `tzlocal`.
  - `Flag` bits: `MISSING`, `OUT_OF_RANGE`, `RATE_OF_CHANGE`, `OUTLIER`, `STUCK`, `MANUAL`, `EXCLUDED`,
    `CORRECTED`, `IMPUTED_SHORT`, `IMPUTED_MODEL`, `FORECAST`, `DERIVED`, `RESAMPLED`, `NO_SAMPLE`. Cleaning sets a
    value to NaN and sets a flag; it never deletes data.
  - `NO_SAMPLE` marks the rows a stream gets only because another input has a sample at that time, when several
    inputs are joined. `align`, `resample` and `interpolate` skip those rows. Before this, an off-grid source
    joined with an on-grid one aligned to the join's padding and came out all NaN.
  - The `SignalError` hierarchy (`SignalConfigError`, `SignalDataError`, `SignalBackendError`). Errors carry a
    dotted `field` path and survive pickling.
  - The step registry with the `flexsignal.steps` entry-point group, and the backend registry
    (`get_backend("scipy:PchipInterpolator")`, lazy imports, "install the extra" errors) with the
    `flexsignal.backends` group.
- **`SignalSpec` (schema `0.1.0`).**
  - Strict, frozen pydantic models with `kind`-discriminated steps, named input slots (`frame`, `parquet`, `csv`)
    and outputs (`parquet`, `csv`).
  - `load_spec` / `dump_spec`, `MIGRATIONS`, `$source` file references, and `spec_hash` over canonical JSON.
  - Conditions (`keep_when`, `invalid_when`) can't run code. `check_condition` parses each one and accepts only
    stream names, numbers, arithmetic, comparisons, `and`/`or`/`not`/`&`/`|`/`~` and the math functions in
    `CONDITION_FUNCTIONS`. It runs when a spec is loaded and before evaluation, and refuses attribute access,
    indexing, strings and other calls with an error naming the field.
  - Errors name the exact field once, e.g. `steps.2.window: '1 hour' is not a duration ...`. A spec with several
    problems lists them all.
  - The JSON Schema is committed at `src/flexsignal/spec/schemas/signal_spec.schema.json`, and a test keeps it
    current.
  - A step *is* its spec: the object built in Python is what JSON loads into. Fitted state is a separate JSON dict
    returned by `fit()`.
- **Steps.**
  - select: `align`, `resample`, `window`, `mask`, `rename`, `combine`, `shift`.
  - clean: `range`, `rate_of_change`, `stuck`, `hampel` (with a `min_scale` noise floor), `tukey` (fences learned
    in `fit`), `expression`.
  - impute: `interpolate` (linear, nearest, previous, PCHIP, Akima via numpy/scipy), limited to gaps no longer
    than `max_gap` and never filling `EXCLUDED` values.
- **Pipelines and entry points.**
  - `Pipeline` / `FittedPipeline` with `fit`, `transform`, `report`, `to_payload` / `from_payload`, and `save` /
    `load` (plain JSON files, no pickle).
  - Every step's effect is recorded in the output's history by comparing its input and output.
  - `flexsignal.run` / `flexsignal.fit` follow the orchestrator contract: spec in, JSON-able `RunResult` out, no
    I/O when data is passed in, and progress callbacks.
- **File I/O.** Parquet that keeps flags, units and provenance in the Arrow metadata, and CSV with exact float
  round-trips.
- **`flexsignal.testing.synthetic`.** A seeded generator of plant-like data (flow, pressure, temperature, pump
  status, an intermittent reference meter). It injects spikes, stuck runs, gaps, out-of-range values, drift and
  steps, with exact ground-truth masks.
- **Tests.**
  - Each cleaner flags exactly the injected faults.
  - Python- and JSON-built pipelines give bit-identical output.
  - Every step kind round-trips through JSON.
  - Orchestrator contract tests: params-model embedding, JSON results and payloads, pickling, no I/O.
  - Every code block in the explanation and how-to docs runs in CI.
- **Docs.**
  - Explanation pages: SignalFrame and flags, the pipeline spec, steps and backends, time and units.
  - How-to guides: clean a stream, save and load a fitted pipeline.
  - An example script.
  - A full API reference, with step, field, flag and backend tables generated from the registries.
- **Tutorials.** Four marimo notebooks in `tutorials/`, from a first look at the data to multi-stream rules:
  a first look, cleaning one stream, pipelines and specs, and several streams. Each scores its steps against
  the injected faults and asserts its own results. CI runs them as scripts, and the docs site publishes them as
  static pages. Install with the new `[tutorials]` extra.

- **Project scaffold (M0).**
  - `flex-signal` distribution with the `flexsignal` import package: a src layout, Python ≥3.11, and the core
    dependencies from `PLAN.md` §5.1.
  - Subpackage skeleton (`core`, `spec`, `io`, `backends`, `steps`, `pipeline`, `eval`, `api`,
    `integrations.flexpse`, `integrations.acquirium`), with each module's purpose and target milestone in its
    docstring.
  - import-linter contracts for the layering, the independence of the integrations, and the rules that only
    `integrations.flexpse` may import flex-pse, pyomo or idaes and only `integrations.acquirium` may import
    acquirium or polars.
  - Tooling: ruff (flex-pse's rules plus `D`, Google style), black, pre-commit with a pre-push suite, and pytest
    tier markers (`unit`, `component`, `integration`). A root `conftest.py` fails collection when a test has
    anything other than exactly one tier marker.
  - CI mirroring flex-pse v0.3.0:
    - uv installs;
    - tests on Python 3.11–3.14 (Linux), plus 3.13 on macOS and Windows;
    - a single coverage run uploaded to Codecov (`codecov.yml`);
    - an integration job;
    - a standard-install check against the built wheel.
  - Sphinx docs (furo, MyST) deployed to GitHub Pages, and a PyPI publish workflow.
  - `CONTRIBUTING.md`, plus issue and PR templates mirroring flex-pse.
- **Test:** `import flexsignal` loads no flex-pse, pyomo, idaes, torch, polars or acquirium (`PLAN.md` D1).
