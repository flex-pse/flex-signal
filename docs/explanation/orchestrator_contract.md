# The orchestrator contract

flex-orchestrator runs studies: it fetches data from a store, runs commands on it, and stores what they return.
flex-signal ships no orchestrator code. Instead its public API keeps a contract that makes wrapping a spec as an
orchestrator command a few lines of code. The contract is tested in `src/flexsignal/tests/contract/`, without
flex-orchestrator installed, and {doc}`tutorial 11 <../tutorials/index>` walks through it.

## The contract

| Promise | What it means |
|---|---|
| A spec goes in, JSON comes out | {func}`~flexsignal.run`, {func}`~flexsignal.fit`, {meth}`FittedPipeline.transform <flexsignal.FittedPipeline.transform>` and {func}`~flexsignal.eval.backtest` take a spec as a `SignalSpec`, a dict, JSON text or a path, and give the same result for each. `RunResult.to_dict()` and `BacktestResult.to_dict()` are JSON |
| No I/O when data is passed | A spec's `inputs` are named slots. The caller binds data to them, `data={"historian": (df, units)}`, and nothing is read or written. Pass `write_outputs=False` to skip the spec's `outputs` too |
| The orchestrator's data shape | Data goes in and comes out as a `DataFrame` with a tz-aware UTC index, plus `units: dict[str, str]`, the shapes `orch.get_data` and `orch.store_data(name, df, units=)` use. `RunResult` has `.data`, `.units`, `.flags`, `.report`, `.spec_hash` and `.provenance` |
| Storage-agnostic fits | {meth}`~flexsignal.FittedPipeline.to_payload` returns `{"spec", "state", "blobs", ...}`, and {meth}`~flexsignal.FittedPipeline.from_payload` rebuilds the fit from it, on any backend |
| Safe on Ray workers | Fitted pipelines, results and errors pickle. Runs are repeatable, and concurrent runs in one process don't affect each other. Optional backends are imported on the worker, when a step needs them |
| Progress and errors | `progress=lambda fraction, message: ...` reports from 0 to 1. Every error is a {class}`~flexsignal.core.SignalError` with a dotted `field` path, such as `steps.1.k`, for the command ledger |
| An embeddable, versioned schema | `SignalSpec` is a pydantic model, so a command's params model can have a `spec: fs.SignalSpec` field. Validating it checks the spec's `schema_version` and migrates an older spec, as {func}`~flexsignal.spec.load_spec` does ({doc}`stability`) |
| Time | flex-signal works in UTC. Naive timestamps are taken as the machine's local time, with a `NaiveTimestampWarning`, so the orchestrator should pass tz-aware data. {class}`~flexsignal.handoff.GridOptions` defaults to `timezone="UTC"`, as the orchestrator does |

## A command, sketched

```python
from pydantic import BaseModel

import flexsignal as fs
from flexsignal.testing import synthetic


class CleanSignalsParams(BaseModel):
    spec: fs.SignalSpec
    output_dataset: str


def clean_signals(params: CleanSignalsParams, data: dict) -> dict:
    result = fs.run(params.spec, data, write_outputs=False)
    # store result.data with result.units under params.output_dataset ...
    return result.to_dict()  # JSON, for the ledger


spec = fs.Pipeline(
    [fs.clean.Range("flow", min=0), fs.impute.Interpolate("flow", max_gap="15min")],
    name="clean",
    inputs=[{"kind": "frame", "name": "historian"}],
).spec
params = CleanSignalsParams.model_validate_json(
    CleanSignalsParams(spec=spec, output_dataset="flow_clean").model_dump_json()
)
signal = synthetic.plant(seed=0, periods=288).signal()
summary = clean_signals(params, {"historian": (signal.values, signal.units)})
assert summary["spec_hash"] == fs.spec_hash(spec)
```

## Things to know when wrapping

- **Pass units with the data.** A bare `DataFrame` bound to a slot has no units, so every stream's unit is `None`.
  Bind `(df, units)` or a {class}`~flexsignal.core.SignalFrame`.
- **The full JSON Schema.** `CleanSignalsParams.model_json_schema()` describes `steps` only as objects with a
  `kind`. {func}`flexsignal.spec.build_json_schema <flexsignal.spec.schema.build_json_schema>` gives every step's options, for editors and study-config
  validation; merge its `$defs` into the study's schema. Methods registered by plugins aren't in it.
- **Blobs are bytes.** A payload's `blobs` maps names to `bytes`, for fitted state that can't be JSON. No step
  uses them yet, so `json.dumps(payload)` works today. A store that writes payloads as JSON should still keep the
  blobs as separate binary objects, so it keeps working when one does.
- **Errors from the params model.** Validating a params model raises pydantic's `ValidationError`, not a
  `SignalError`. For an error whose `field` names the step by index (`steps.1.k`), load the spec with
  {func}`~flexsignal.spec.load_spec` or let {func}`~flexsignal.run` do it.
- **Cache on the spec hash.** {func}`~flexsignal.spec.spec_hash` is the same however the spec's keys are ordered and
  whether defaults are written. It is frozen with the schema, so it works as a cache key across releases.
- **Registered methods.** A spec that names a registered method needs it registered on the worker too: pass
  `plugins=[...]` to `fs.run`, `fs.fit` or `FittedPipeline.load`, or install the package that registers it.
