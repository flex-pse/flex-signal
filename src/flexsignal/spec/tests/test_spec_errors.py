import gzip
import json
from typing import Literal

import pytest
from pydantic import TypeAdapter, ValidationError

import flexsignal as fs
from flexsignal.core import SignalConfigError
from flexsignal.spec import (
    SCHEMA_VERSION,
    SignalSpec,
    SignedDuration,
    StepSpec,
    TimestampStr,
    load_spec,
)
from flexsignal.spec.schema import export_json_schema, schema_text
from flexsignal.testing import synthetic


def _hampel():
    return {"kind": "hampel", "streams": ["flow"], "window": "1h"}


def _spec_dict(**kw):
    return {"schema_version": SCHEMA_VERSION, "name": "t", **kw}


# -- schema export ---------------------------------------------------------------


@pytest.mark.unit
def test_export_json_schema_writes_the_committed_text(tmp_path):
    path = export_json_schema(tmp_path / "nested" / "dir" / "schema.json")
    assert path == tmp_path / "nested" / "dir" / "schema.json"
    assert path.read_bytes().decode("utf-8") == schema_text()
    assert "\r\n" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["title"] == "SignalSpec"


# -- load_spec errors ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("version", ["1.0", "a.b.c", 1])
def test_a_malformed_schema_version_is_refused(version):
    with pytest.raises(SignalConfigError, match="must look like X.Y.Z") as exc:
        load_spec({"schema_version": version, "name": "t"})
    assert exc.value.field == "schema_version"
    assert exc.value.value == version


@pytest.mark.unit
def test_source_references_read_gzipped_json(tmp_path):
    with gzip.open(tmp_path / "steps.json.gz", "wt", encoding="utf-8") as handle:
        json.dump([_hampel()], handle)
    spec = load_spec(_spec_dict(steps={"$source": "steps.json.gz"}), base_dir=tmp_path)
    assert [s.kind for s in spec.steps] == ["hampel"]


@pytest.mark.unit
def test_a_missing_source_file_is_named(tmp_path):
    with pytest.raises(SignalConfigError, match="file not found") as exc:
        load_spec(_spec_dict(steps={"$source": "nope.json"}), base_dir=tmp_path)
    assert "nope.json" in exc.value.message
    with pytest.raises(SignalConfigError, match="file not found"):
        load_spec(tmp_path / "missing_spec.json")


@pytest.mark.unit
def test_a_source_file_that_is_not_json_is_named(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SignalConfigError, match="bad.json is not valid JSON"):
        load_spec(_spec_dict(steps={"$source": "bad.json"}), base_dir=tmp_path)


@pytest.mark.unit
def test_a_spec_string_that_is_not_json_is_refused():
    with pytest.raises(SignalConfigError, match="the spec is not valid JSON"):
        load_spec('{"name": ')


@pytest.mark.unit
def test_a_spec_file_must_hold_an_object(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SignalConfigError, match="a spec must be a JSON object"):
        load_spec(path)


@pytest.mark.unit
def test_steps_must_be_a_list():
    with pytest.raises(SignalConfigError, match="steps must be a list") as exc:
        load_spec(_spec_dict(steps={"kind": "hampel"}))
    assert exc.value.field == "steps"


@pytest.mark.unit
@pytest.mark.parametrize("raw", [5, {"stream": "flow"}, {"kind": 3}])
def test_each_step_needs_a_kind(raw):
    with pytest.raises(
        SignalConfigError, match="must be an object with a 'kind'"
    ) as exc:
        load_spec(_spec_dict(steps=[_hampel(), raw]))
    assert exc.value.field == "steps.1.kind"


# -- spec models ------------------------------------------------------------------


@pytest.mark.unit
def test_signed_durations_may_be_negative_but_must_parse():
    adapter = TypeAdapter(SignedDuration)
    assert adapter.validate_python("-5min") == "-5min"
    with pytest.raises(ValidationError, match="'soon' is not a duration"):
        adapter.validate_python("soon")


@pytest.mark.unit
def test_timestamps_must_parse():
    adapter = TypeAdapter(TimestampStr)
    assert adapter.validate_python("2025-01-01T00:00:00Z") == "2025-01-01T00:00:00Z"
    with pytest.raises(ValidationError, match="'yesterday-ish' is not a timestamp"):
        adapter.validate_python("yesterday-ish")


class _NoTransform(StepSpec):
    kind: Literal["no_transform"] = "no_transform"


@pytest.mark.unit
def test_a_step_without_transform_says_so():
    sf = synthetic.plant(seed=1, periods=10).signal()
    step = _NoTransform()
    assert step.fit(sf) == {}
    with pytest.raises(NotImplementedError, match="'no_transform' does not implement"):
        step.transform(sf, {})


@pytest.mark.unit
def test_steps_none_means_no_steps():
    assert SignalSpec(name="t", steps=None).steps == ()


@pytest.mark.unit
@pytest.mark.parametrize("steps", [fs.clean.Hampel("flow", window="1h"), {}, 5])
def test_steps_must_be_a_list_of_steps(steps):
    with pytest.raises(ValidationError, match="steps must be a list of steps"):
        SignalSpec(name="t", steps=steps)


@pytest.mark.unit
def test_a_step_dict_needs_a_kind():
    with pytest.raises(ValidationError, match="step 1 needs a 'kind'"):
        SignalSpec(name="t", steps=[_hampel(), {"stream": "flow"}])


@pytest.mark.unit
def test_a_step_must_be_a_step_object_or_a_dict():
    with pytest.raises(ValidationError, match="step 0 must be a step object or a dict"):
        SignalSpec(name="t", steps=[3.5])


@pytest.mark.unit
def test_dict_steps_resolve_through_the_registry():
    spec = SignalSpec(name="t", steps=[_hampel()])
    assert spec.steps == (fs.clean.Hampel("flow", window="1h"),)


@pytest.mark.unit
def test_input_slot_names_must_be_unique():
    inputs = [{"kind": "frame", "name": "a"}, {"kind": "frame", "name": "a"}]
    with pytest.raises(ValidationError, match=r"input slot names must be unique.*'a'"):
        SignalSpec(name="t", inputs=inputs)
    with pytest.raises(SignalConfigError, match="input slot names must be unique"):
        load_spec(_spec_dict(inputs=inputs))
