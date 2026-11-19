# flex-orchestrator ↔ flex-signal: what moves, what stays

**Purpose.** flex-orchestrator and flex-signal both turn timeseries into model inputs and fit data today. This
document records the agreed split (flex-signal PLAN.md, D22) and is the input to flex-orchestrator's refactor
plan. After the refactor, flex-orchestrator depends on flex-signal for the data work and keeps orchestration.

**Status.** flex-signal's side is built (M3, to be tagged 0.3.0): `flexsignal.handoff` and
`flexsignal.integrations.flexpse` pass flex-orchestrator's ported tests (section 4). flex-orchestrator switches to
them in its own refactor (section 5). Inventory taken from flex-orchestrator branch `multi-time-block`
(`f889227`, 2026-11-12), which pins flex-pse 0.3.1.

---

## 1. The rule

> **Data in, data out → flex-signal.** Code that takes timeseries and returns values, prepared frames or model
> elements, with no store, no study and no Ray, lives in flex-signal, so a plain script can call it.
>
> **Where data lives and how work runs → flex-orchestrator.** Stores, selectors, studies, sweeps, variants,
> tags, fit-run records, Ray, the CLI and service, export and archive stay in flex-orchestrator, which calls
> flex-signal for the data work.

Two consequences:

- **Regressors stay out of flex-signal.** flex-signal stops at a prepared `X, y`; a script calls flex-pse's
  regressor directly. flex-orchestrator keeps its regressor wrappers (`fitting/regress.py`,
  `flexpse/regression.py`) because sweeps, studies and automated fits need a regressor named by a string.
- **Dependency direction.** flex-orchestrator imports flex-signal; flex-signal never imports flex-orchestrator.
  flex-signal imports flex-pse only inside `flexsignal.integrations.flexpse`, and only inside functions.

## 2. Inventory

### Moves to flex-signal

| flex-orchestrator today | What it does | flex-signal home (M3) |
|---|---|---|
| `inputs/resolve.py`: `_read` (window part), `_years`, `_utc`, `_MARGIN` | Works out which UTC window to read for a model horizon: `offset`, `data_years` (calendar dates in another year), a 1-day margin each side (none for `reduce`), local → UTC with "later time" for DST gaps and repeats, a backward step reading one step earlier | `flexsignal.handoff.data_window(grid, options)` |
| `inputs/resolve.py`: `_read` (index part), `_series`, `_split`, `_reduce`, `_step` | Turns data into one value per model time step: UTC → facility wall clock, shift back by `offset` / `data_years`, resample `forward` `[t, t+dt)` or `backward` `(t-dt, t]` with an `agg`, split coarser summed amounts evenly over whole steps, gap rules `interpolate` (in time, inside only, `limit` steps) / `ffill` (`limit`) / `zero` / `error`, or `reduce` the horizon to one number | `flexsignal.handoff.to_grid(data, grid, options) -> GridValues` |
| `inputs/resolve.py`: unit formatting, `_hash`, `filled` counts | `{"value", "units"}` with `**` written `^`; a sha256 of the values; how many steps were filled | `GridValues.values`, `.units`, `.filled`, `.data_hash(rename)`, `.to_overrides(rename)`; `handoff.values_hash` |
| `inputs/models.py`: `InputResample`, the alignment fields of `ModelInput` (`timezone`, `resample`, `gaps`, `limit`, `reduce`, `offset`, `data_years`) and their validators | The options above, as JSON | `flexsignal.handoff.GridOptions` (same field names, defaults and validators) |
| `inputs/scopes.py`: `grids_of`, `scope_of`, `targets`, `_wildcard` | Which time block a model path belongs to; `*` in a scope's index; a path under no scope set in every scope | `flexsignal.handoff` (pure path logic, no model import) |
| `fitting/prep.py`: `prepare` (minus selector provenance), `_gaps`, `data_hash`, `split_holdout` | Join input and output frames (`inner`/`outer`), `max_rows` guard, optional resample, gap rules `error` / `drop` / `interpolate` (+`limit`) / `longest_segment`, a step log, the data hash, the hold-out split | `flexsignal.handoff.fit_data(X, y, prep) -> FitData`, `split_holdout`, `data_hash` |
| `fitting/models.py`: `Prep`, `Resample`, `Validation` | Those options as JSON | `flexsignal.handoff.FitPrep` (same names, defaults, validators); `holdout` stays a string argument |
| `fitting/regress.py`: `metrics` | RMSE, MAE, MAPE over non-zero actuals | `flexsignal.eval.metrics` (module `flexsignal.eval.scores`) |
| `flexpse/timegrid.py`: `time_grid` (flowsheet specs and built models), `time_scopes`, `_of_spec`, `_of_model`, `_time_block` | A flowsheet spec's or built model's time points, one axis per scope | `flexsignal.integrations.flexpse.time_grid`, `time_scopes` |
| `flexpse/build.py`: `with_overrides`, `_set_field` (costing part), `apply_values` | Values → named `fix` elements (`override:<path>`); a `costing` field path edits that field instead; fix a built model's variables, grouped by time scope, via `flexops.apply_spec` | `flexsignal.integrations.flexpse.apply_values` (a spec or a built model) |

### Stays in flex-orchestrator

| Module | Why it stays |
|---|---|
| `fitting/selectors.py`, the selector half of `ModelInput` (`dataset`, `result`, `ref_uris`, `streams`, `resolve`, `sparql`, `start`, `end`, `units`, `rename`) | Reading the store; unit conversion on read |
| `inputs/resolve.py`: `resolve`, `_layer`, `_entry`, `_probe`, `_load`, the read cache, `Resolved` and its provenance records | Per-entry orchestration: which entries, swept entries replacing base ones, "two entries set the same path", reading each selection once, provenance records. It calls flex-signal for each window and series |
| `fitting/command.py`, `FitParams`, `Variant`, `FitPlan`, `FitRecord`, `FitSummary`, `fitting/store.py` | The `fit_surrogate` command: variants, tags, fit runs, storage, Ray |
| `fitting/regress.py` (minus `metrics`), `flexpse/regression.py` | Regressors named by string, for automated fits |
| `flexpse/timegrid.py`: `_of_design`, `horizon_scope`; `flexpse/design.py` | Design specs need `flexdesign` and flex-orchestrator's design loader. The hook maps each horizon to its scope and calls flex-signal's `time_grid` on that horizon's flowsheet spec |
| `flexpse/build.py`: `build`, `configure`, `_configure_design`, `refuse_design_model` | Building and configuring models; the build hook uses flex-signal's `apply_values` on the spec |
| `hooks.py` and the rest | Hooks keep flex-orchestrator model-library agnostic; the flex-pse hooks become thin calls into flex-signal |

## 3. The flex-signal API flex-orchestrator will call

These are the names as built in M3.

```python
from flexsignal import handoff
from flexsignal.integrations import flexpse as fsp

# model_inputs, per entry and scope
options = handoff.GridOptions(timezone="America/Los_Angeles", gaps="error")  # ModelInput's alignment fields
start_utc, end_utc = handoff.data_window(grid, options)                       # what to read from the store
values = handoff.to_grid(frame, grid, options)                                # frame: DataFrame, SignalFrame or (df, units)
values.values, values.units, values.filled, values.estimated, values.window
overrides = values.to_overrides(rename={"flow": "plant.pump.flow_in"})        # {path: [..] | {"value", "units"}}
values.data_hash(rename={"flow": "plant.pump.flow_in"})                       # == resolve's record["data_hash"]

handoff.grids_of(fsp.time_grid(spec_or_model))                                # {scope: DatetimeIndex | None}
handoff.targets("plant.pump.flow_in", scopes)                                  # [(scope, path)]

spec = fsp.apply_values(spec, overrides)                                      # a spec: a new spec, to build
fsp.apply_values(model, overrides)                                            # after build

# fit_surrogate, per fit
fit = handoff.fit_data([x_frame, ...], [y_frame], handoff.FitPrep(gaps="drop", resample={"freq": "1h"}))
fit.X, fit.y, fit.units, fit.steps, fit.data_hash, fit.input_units, fit.output_units
X_train, y_train, X_test, y_test = handoff.split_holdout(fit.X, fit.y, "7D")
scores = fs.eval.metrics(actual, predicted)
```

**Errors.** flex-signal raises `SignalDataError` / `SignalConfigError` with the column and the reason.
flex-orchestrator catches them and re-raises its own `ModelInputError` / `FitError`, prefixing the entry
(`Model input {'dataset': ...}`) as now, so its users see the same messages.

**Differences found while porting** (everything else is unchanged, and the ported tests pin it):

- `to_grid` clips its input to `data_window` itself, so passing more data than the window gives the same result as
  flex-orchestrator's store read did. flex-orchestrator can keep reading just the window.
- With no data in the window, `to_grid` raises `SignalDataError` ("no samples in the window ..."). flex-orchestrator's
  selectors raise "matched no rows" before that point, so its messages don't change.
- `to_grid` takes the model's grid one time axis at a time; flex-orchestrator keeps looping over scopes.
- Plain DataFrames with naive timestamps: `to_grid` reads them as this machine's local time with a warning (flex-signal
  D14); `fit_data` keeps them as they are, as flex-orchestrator's fit prep does. Store data is tz-aware UTC, so
  neither case arises from the store.
- `fit_data` concatenates with `sort=True` explicitly (pandas 3 warns otherwise); the result is sorted as before.
- One function, `apply_values`, replaces `with_overrides` and `apply_values`: given a spec it returns the new spec
  (what `with_overrides` did), given a built model it fixes it in place and returns it. On a built model it also sets
  mutable Params (a price built as a Param); a `costing.energy_prices...` path on a built model raises, because
  flex-pse builds a spec's flat price as a constant.

**Option models.** `GridOptions` and `FitPrep` are pydantic models with flex-orchestrator's exact field names,
defaults and validation. flex-orchestrator can build `ModelInput` from `Selector` plus `GridOptions`, and use
`FitPrep` as `Prep`, so study configs, the CLI and the HTTP API don't change and validation lives in one place.

**Flags.** When the input is a flex-signal `SignalFrame`, samples flagged `EXCLUDED` or `NO_SAMPLE` are not
samples, and `GridValues` reports how many model steps rest on filled (`IMPUTED_*`) or corrected values. Plain
DataFrames from the store have no flags, so their results are unchanged.

## 4. Behaviour that must not change

flex-signal ports these flex-orchestrator tests, as pure-data tests with the same inputs and expected values.
flex-orchestrator keeps its own versions running against the store throughout the switch.

**`inputs/tests/test_inputs.py`** — move:
`test_hourly_data_fills_every_step`, `test_a_column_without_a_unit_is_a_plain_list`,
`test_the_facility_time_zone`, `test_daylight_saving_days`, `test_resample_aggregates_onto_a_coarser_step`,
`test_a_backward_step_takes_the_interval_ending_at_it`, `test_a_sum_of_coarser_data_is_split_over_the_steps`,
`test_a_sum_splits_only_over_whole_steps`, `test_gap_rules`, `test_gaps_that_cant_be_filled`,
`test_reduce_gives_one_number`, `test_a_model_without_time_steps_takes_only_reduced_inputs`,
`test_offset_reads_another_window`, `test_data_years_read_another_years_dates`,
`test_a_path_belongs_to_the_nearest_time_block_above_it`, `test_each_path_is_lined_up_with_its_own_time_block`,
`test_a_path_under_no_time_block_is_set_in_each`.
Stay: `test_units_convert`, `test_paths_set_once_per_list_and_swept_entries_replace`,
`test_a_selector_that_matches_nothing`, `test_each_distinct_selection_is_read_once`, `test_invalid_entries`
(selector half), `test_an_entry_reads_only_the_time_blocks_it_sets`.

**`inputs/tests/test_inputs_flexpse.py`** — move `test_time_grid_matches_the_built_model`; the study tests stay.

**`fitting/tests/test_fitting.py`** — move `test_prepare_aligns_and_hashes`, `test_gap_handling`,
`test_interpolation_leaves_ends_and_long_gaps`, `test_resample_and_guards`, `test_holdout_split`,
`test_metrics`.

**`flexpse/tests`** — move `test_with_overrides_adds_named_fix_elements_without_mutating` and
`test_overrides_edit_costing_fields`; the build tests stay.

Details easy to lose in a port, each covered by the tests above:

- Forward steps `[t, t+dt)` are labelled `t`; backward `(t-dt, t]` are labelled `t`, and a backward window
  reads one step earlier.
- A summed amount coarser than the step is split into equal parts over the steps its interval covers, and only
  over a whole number of steps; otherwise it is an error.
- `interpolate` is in time, fills inside only (never the ends), and `limit` counts steps; `ffill` does fill the
  end.
- The read window has a 1-day margin each side (none for `reduce`) so end gaps can be interpolated.
- `data_years` moves calendar dates with `DateOffset(years=...)`; a modelled 29 February reads 28 February in a
  year without one.
- Local times skipped or repeated by DST map to the later instant when converted to UTC.
- The data hash is sha256 of sorted, compact JSON of the values (resolve) or of time, X and y with floats in
  shortest round-trip form (prep); hashes stay identical, so stored provenance stays comparable.
- Units: `**` becomes `^` in `{"value", "units"}`.
- `apply_values` on a spec names its elements `override:<path>`, doesn't mutate the input spec, and edits a `costing`
  field when the path's head names a costing element and its next part is one of its fields.

## 5. Migration steps

**flex-signal (M3, 0.3.0)**

1. Port the tests listed in section 4, then implement `flexsignal.handoff` and
   `flexsignal.integrations.flexpse` until they pass.
2. Tag 0.3.0. Pin flex-pse 0.3.1 in flex-signal's CI and `environment.yml`.
3. Tutorial 07 and the docs use the same functions in a plain script (clean → `to_grid` → `apply_values` →
   solve; `fit_data` → flex-pse regressor).

**flex-orchestrator (its refactor)**

1. Add the flex-signal dependency, pinned by git tag (see decision A below).
2. Replace `inputs/models.py`'s alignment fields with `GridOptions`, and `Prep` / `Resample` with `FitPrep`;
   keep the field names so study configs and schemas don't change. Regenerate the committed schemas and check
   the diff is empty apart from descriptions.
3. Make `resolve._read` / `_series` / `_reduce` call `data_window` and `to_grid`; keep `resolve`, `_layer`,
   `_entry`, the cache and provenance.
4. Make `inputs/scopes.py` re-export flex-signal's `grids_of`, `scope_of`, `targets` (or call them directly).
5. Make `fitting/prep.py` call `fit_data`, `split_holdout`, `data_hash`, and `regress.metrics` call
   `flexsignal.eval.metrics`; keep `Prepared`'s selector provenance.
6. Make the flex-pse hooks call flex-signal: `time_grid` (flowsheet specs and built models; design specs map
   horizons to scopes and call it per horizon), `with_overrides` and `apply_values` → `apply_values`.
7. Delete the moved private functions and their tests once the store-backed tests pass unchanged.
8. Check stored provenance hashes from before and after the switch match on the same data.

## 6. Decisions still open

| # | Question | Leaning |
|---|---|---|
| A | flex-signal as a **core** dependency of flex-orchestrator, or behind the `flexpse` extra? Core means pandas ≥ 3 for every install, plus scipy, scikit-learn, statsmodels and pyarrow (and pandera, which flex-signal could drop). `model_inputs` and `fit_surrogate` also run on the memory backend without flex-pse | Core, since `model_inputs` and fit prep are core features; flex-signal drops pandera first |
| B | Default `timezone` | Keep `"UTC"`, as now. (flex-signal reads *naive input data* as local time, D14, but a model's zone is a deliberate choice and must not depend on the machine) |
| C | Design-spec grids (`_of_design`) | Stay in flex-orchestrator, calling flex-signal per horizon (section 2) |
| D | Where the costing-field editing lives | flex-signal, in `apply_values` on a spec; flex-orchestrator's `configure` (any spec field by path) stays |
| E | flex-orchestrator `limit` counts steps; flex-signal `interpolate` uses `max_gap` durations | `GridOptions` / `FitPrep` keep `limit` in steps for compatibility; flex-signal's own steps keep durations |
