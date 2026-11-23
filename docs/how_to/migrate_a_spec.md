# Upgrade a spec to a newer release

Specs saved by an older flex-signal load in a newer one: {func}`~flexsignal.spec.load_spec` reads the spec's
`schema_version` and upgrades it through {data}`~flexsignal.spec.io.MIGRATIONS`. You don't need to do anything to run
an old spec. This page shows how to write it out at the new version, what the errors mean, and how a contributor
adds a migration.

0.1.0 is the first released schema, so there are no migrations yet. The steps below work the same once there are.

## Rewrite a stored spec at the current version

Load it, then dump it. The dumped spec has the current `schema_version`:

```python
import flexsignal as fs

old = {
    "schema_version": "0.1.0",
    "name": "flow_cleanup",
    "steps": [{"kind": "range", "streams": ["flow"], "min": 0}],
}
spec = fs.load_spec(old)  # migrated, if it is older than this release
upgraded = fs.dump_spec(spec)
assert upgraded["schema_version"] == fs.SCHEMA_VERSION
```

Pass a path instead of a dict to read a file, and `fs.dump_spec(spec, "spec.json")` to write one. A spec embedded in
another pydantic model, such as an orchestrator command's params, is checked and migrated the same way when the
model is validated.

## A spec from a newer release

An older flex-signal can't know what a newer schema means, so it refuses the spec:

```python
import flexsignal as fs

newer = {"schema_version": "99.0.0", "name": "from_the_future", "steps": []}
try:
    fs.load_spec(newer)
except fs.SignalConfigError as err:
    assert err.field == "schema_version"
    print(err)  # ... newer than this flex-signal release supports (...). Upgrade flex-signal.
```

Upgrade flex-signal on that machine. `pip install -U flex-signal`.

## A fitted pipeline from an older release

{meth}`~flexsignal.FittedPipeline.from_payload` and {meth}`~flexsignal.FittedPipeline.load` migrate the saved
spec the same way. If the payload format itself has changed and can't be read, the error says to refit: run
{func}`~flexsignal.fit` on the spec and the training data again.

## Add a migration (contributors)

When a change to the schema needs one (see {doc}`../explanation/stability`), add a function that takes a spec dict
at the old version and returns it at the next:

```python
def rename_hampel_window(data):
    """0.1.0 -> 0.2.0: an example rename of hampel's option ``window`` to ``span``."""

    def upgrade(step):
        if step.get("kind") != "hampel":
            return step
        return {("span" if k == "window" else k): v for k, v in step.items()}

    data = dict(data)
    data["steps"] = [upgrade(s) for s in data.get("steps", [])]
    data["schema_version"] = "0.2.0"
    return data
```

It is registered under the version it upgrades from, `MIGRATIONS["0.1.0"] = rename_hampel_window`, in
`flexsignal/spec/io.py`, and `SCHEMA_VERSION` becomes `"0.2.0"`. Then freeze the new version with
`python -m flexsignal.spec.tests.freeze`; the frozen-spec tests check that the 0.1.0 spec still loads through the
new migration.
