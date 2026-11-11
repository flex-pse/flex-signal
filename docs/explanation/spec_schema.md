# The pipeline spec

A {class}`~flexsignal.spec.SignalSpec` is a declarative, versioned description of a pipeline. You can store it,
diff it, and hand it to another tool, which will re-run it and get the same result. Its conventions mirror flex-pse's
`FlowsheetSpec`.

## A step is its own spec

Every step class is a frozen pydantic model. `fs.clean.Hampel("flow", window="1h")` is both the step you run and
its configuration, so a pipeline built in Python and the same pipeline loaded from JSON are *the same objects*,
and they give bit-identical results. The test suite checks this.

```python
import json
import flexsignal as fs

pipe = fs.Pipeline(
    [fs.clean.Range("flow", min=0), fs.impute.Interpolate("flow", max_gap="15min")],
    name="flow_cleanup",
)
text = json.dumps(fs.dump_spec(pipe.spec), indent=2)
assert fs.load_spec(text) == pipe.spec
```

The JSON looks like this. Fields left at their default are omitted, except `schema_version` and each `kind`:

```json
{
  "schema_version": "0.1.0",
  "name": "flow_cleanup",
  "steps": [
    {"kind": "range", "streams": ["flow"], "min": 0.0},
    {"kind": "interpolate", "streams": ["flow"], "max_gap": "15min"}
  ]
}
```

## Fitted state lives next to the spec, never inside it

Some steps learn from training data. For example, `tukey` learns quartiles. A step's `fit(sf)` returns that
state as a plain JSON dict, and `transform(sf, state)` takes it back. The spec never changes when you fit it. A
{class}`~flexsignal.FittedPipeline` is the spec plus one state per step, and saves as plain JSON, with no pickle
(see [Save and load a fitted pipeline](../how_to/save_and_load_a_fitted_pipeline.md)).

## Inputs and outputs

`inputs` declares named **slots**. A `frame` slot is filled by the caller at run time:
`fs.run(spec, {"historian": df})`. A `parquet` or `csv` slot names a file, which is read only when the caller
doesn't bind data to that slot. Several slots are joined on time. The rows the join adds to a stream, at times
when only another input has a sample, are flagged `MISSING` and `NO_SAMPLE`, so start such a pipeline with an
`align` step to put every stream on one grid. With no `inputs` at all, the pipeline takes one frame directly.

`outputs` lists files that {func}`flexsignal.run` writes. Pass `write_outputs=False` to keep a run free of file
I/O.

## Versions, migrations and errors

- `schema_version` is required in JSON. {func}`~flexsignal.spec.load_spec` refuses specs newer than the installed
  release, and upgrades older ones through `flexsignal.spec.MIGRATIONS`.
- Errors say which field is wrong, e.g. `steps.2.window: '15 minutes' is not a duration`. They carry the same
  path in `error.field`, so tools such as flex-orchestrator can point at it.
- `{"$source": "file.json"}` anywhere in a spec is replaced by that file's contents. The path is relative to the
  spec file. Use it to keep large tables out of the spec.
- {func}`~flexsignal.spec.spec_hash` is the sha256 of the canonical JSON. It is the same however keys are ordered,
  and whether or not defaults are written out. It is stamped into every provenance record.

## Step reference

Every registered step:

```{eval-rst}
.. flexsignal-step-table::
```

The full JSON Schema, generated from these models, is committed with the package:
{download}`signal_spec.schema.json <../../src/flexsignal/spec/schemas/signal_spec.schema.json>`. Editors can use
it to validate and autocomplete specs.

## Conditions can't run code

Conditions in a spec (`keep_when`, `invalid_when`) can only read streams. When a spec is loaded, and again
before a condition is evaluated, flex-signal parses it and accepts only:

- stream names (in backticks if they aren't Python identifiers), numbers, `True` and `False`;
- arithmetic (`+ - * / // % **`; the exponent after `**` must be a number no larger than 100);
- comparisons (`== != < <= > >=`, chained like `0 < flow <= 50`) and `and or not & | ~`;
- the math functions in {data}`~flexsignal.core.CONDITION_FUNCTIONS`, such as `abs(x)` and `sqrt(x)`.

Anything else is refused with an error naming the spec field, before pandas sees the expression. That covers
attribute access (`flow.__class__`), indexing, strings, other functions, keyword arguments, `@` variables and
`lambda`. pandas `eval` is not a security boundary on its own; this check is.

```{warning}
Conditions are safe, but a spec still names **files**: `$source` references, `csv` and `parquet` inputs, and
outputs. A spec can read or write any file the process can. Before running a spec from someone else, check its
paths, or run it where it can only reach the files it should.
```
