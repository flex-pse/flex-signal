import json

import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.core import SignalConfigError, registered_steps
from flexsignal.spec import (
    MIGRATIONS,
    SCHEMA_VERSION,
    SignalSpec,
    dump_spec,
    load_spec,
    spec_hash,
)
from flexsignal.spec.schema import SCHEMA_PATH, schema_text
from flexsignal.testing.examples import one_of_every_step


def _spec(steps=None, **kw):
    return SignalSpec(
        name="t", steps=steps if steps is not None else one_of_every_step(), **kw
    )


@pytest.mark.unit
def test_examples_cover_every_registered_step_kind():
    covered = {s.kind for s in one_of_every_step()}
    assert covered == set(
        registered_steps()
    ), "add an example of each new step kind to flexsignal.testing.examples"


@pytest.mark.unit
@pytest.mark.parametrize("step", one_of_every_step(), ids=lambda s: s.kind)
def test_every_step_kind_round_trips(step):
    spec = _spec([step])
    text = json.dumps(dump_spec(spec))
    assert load_spec(text) == spec


@pytest.mark.unit
def test_full_spec_round_trips_with_inputs_outputs_metadata(tmp_path):
    spec = _spec(
        inputs=[{"kind": "frame", "name": "historian"}],
        outputs=[{"kind": "csv", "path": "out.csv", "include_flags": True}],
        metadata={"site": "demo", "n": 3},
        description="all of it",
    )
    path = tmp_path / "spec.json"
    dump_spec(spec, path)
    assert load_spec(path) == spec
    assert spec_hash(load_spec(path)) == spec_hash(spec)


@pytest.mark.unit
def test_dump_keeps_kind_and_version_but_drops_defaults():
    data = dump_spec(_spec([fs.clean.Hampel("flow", window="1h")]))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["steps"] == [{"kind": "hampel", "streams": ["flow"], "window": "1h"}]


@pytest.mark.unit
def test_hash_ignores_key_order_and_explicit_defaults():
    a = load_spec(
        {
            "schema_version": SCHEMA_VERSION,
            "name": "t",
            "steps": [{"kind": "hampel", "streams": ["f"], "window": "1h"}],
        }
    )
    b = load_spec(
        {
            "name": "t",
            "schema_version": SCHEMA_VERSION,
            "steps": [
                {"window": "1h", "n_sigma": 3.0, "streams": ["f"], "kind": "hampel"}
            ],
        }
    )
    assert spec_hash(a) == spec_hash(b)


@pytest.mark.unit
def test_errors_name_the_field():
    base = {"schema_version": SCHEMA_VERSION, "name": "t"}
    with pytest.raises(SignalConfigError) as exc:
        load_spec(
            {
                **base,
                "steps": [
                    {"kind": "range", "streams": ["f"], "min": 0},
                    {"kind": "hampel", "streams": ["f"], "window": "soon"},
                ],
            }
        )
    assert exc.value.field == "steps.1.window"
    with pytest.raises(SignalConfigError) as exc:
        load_spec({**base, "steps": [{"kind": "no_such_step"}]})
    assert exc.value.field == "steps.0.kind" and "Known kinds" in exc.value.message
    with pytest.raises(SignalConfigError) as exc:
        load_spec(
            {
                **base,
                "steps": [
                    {"kind": "hampel", "streams": ["f"], "window": "1h", "typo": 1}
                ],
            }
        )
    assert exc.value.field == "steps.0.typo"
    with pytest.raises(SignalConfigError, match="no schema_version"):
        load_spec({"name": "t", "steps": []})
    with pytest.raises(SignalConfigError, match="newer than"):
        load_spec({**base, "schema_version": "99.0.0"})


@pytest.mark.unit
def test_error_message_names_the_field_once():
    base = {"schema_version": SCHEMA_VERSION, "name": "t"}
    step = {"kind": "hampel", "streams": ["f"], "window": "soon"}
    with pytest.raises(SignalConfigError) as exc:
        load_spec({**base, "steps": [step]})
    assert str(exc.value).count("steps.0.window") == 1
    assert "Value error" not in str(exc.value)
    with pytest.raises(SignalConfigError) as exc:
        load_spec({**base, "steps": [{**step, "n_sigma": -1}]})
    assert str(exc.value).startswith("steps.0.window: 2 problems:")
    assert "steps.0.n_sigma" in str(exc.value)


@pytest.mark.unit
def test_duplicate_step_names_are_rejected():
    with pytest.raises(SignalConfigError, match="unique"):
        load_spec(
            {
                "schema_version": SCHEMA_VERSION,
                "name": "t",
                "steps": [
                    {"kind": "range", "streams": ["f"], "min": 0, "name": "x"},
                    {"kind": "range", "streams": ["f"], "max": 9, "name": "x"},
                ],
            }
        )


@pytest.mark.unit
def test_migration_hook_runs(monkeypatch):
    def rename_title_to_name(data):
        data = dict(data)
        data["name"] = data.pop("title")
        data["schema_version"] = SCHEMA_VERSION
        return data

    monkeypatch.setitem(MIGRATIONS, "0.0.9", rename_title_to_name)
    spec = load_spec({"schema_version": "0.0.9", "title": "migrated", "steps": []})
    assert spec.name == "migrated"
    with pytest.raises(SignalConfigError, match="no migration"):
        load_spec({"schema_version": "0.0.1", "name": "t"})


@pytest.mark.unit
def test_source_references_resolve_relative_to_the_spec(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "mapping.json").write_text(json.dumps({"FT_101": "flow"}))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "name": "t",
                "steps": [
                    {"kind": "rename", "mapping": {"$source": "sub/mapping.json"}}
                ],
            }
        )
    )
    assert load_spec(spec_path).steps[0].mapping == {"FT_101": "flow"}


@pytest.mark.unit
def test_steps_are_frozen_and_accept_positional_streams():
    step = fs.clean.Range("flow", min=0)
    assert step.streams == ("flow",)
    with pytest.raises(ValidationError):
        step.min = 5
    with pytest.raises(TypeError):
        fs.clean.Range("flow", streams=["x"], min=0)


@pytest.mark.unit
def test_committed_json_schema_is_current():
    assert SCHEMA_PATH.read_text(encoding="utf-8") == schema_text(), (
        "the committed JSON Schema is stale; regenerate it with "
        "`python -m flexsignal.spec.schema`"
    )
    schema = json.loads(schema_text())
    assert set(schema["$defs"]) >= {cls.__name__ for cls in registered_steps().values()}
