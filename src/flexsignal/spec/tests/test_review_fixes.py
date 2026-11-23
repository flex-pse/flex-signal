"""Regression tests for review findings in the spec layer and the step registry."""

import json
import types
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

import flexsignal as fs
from flexsignal.core import SignalConfigError, SignalWarning, registry
from flexsignal.core.registry import get_step_class, load_entry_points
from flexsignal.spec import (
    SCHEMA_VERSION,
    Duration,
    SignalSpec,
    SignedDuration,
    StepSpec,
    TimestampStr,
    load_spec,
)


def _spec(**extra):
    return {"schema_version": SCHEMA_VERSION, "name": "x", **extra}


# -- durations and timestamps --------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("model", [Duration, SignedDuration])
def test_nat_is_not_a_duration(model):
    with pytest.raises(ValidationError, match="'NaT' is not a duration"):
        TypeAdapter(model).validate_python("NaT")


@pytest.mark.unit
@pytest.mark.parametrize("text", ["NaT", "now", "today", "yesterday", "tomorrow"])
def test_nat_and_relative_words_are_not_timestamps(text):
    with pytest.raises(ValidationError):
        TypeAdapter(TimestampStr).validate_python(text)


@pytest.mark.unit
def test_a_step_with_a_nat_duration_is_refused_on_load():
    spec = _spec(steps=[{"kind": "stuck", "streams": ["flow"], "min_duration": "NaT"}])
    with pytest.raises(SignalConfigError, match="not a duration") as exc:
        load_spec(spec)
    assert exc.value.field == "steps.0.min_duration"


@pytest.mark.unit
def test_a_step_starting_now_is_refused_on_load():
    with pytest.raises(SignalConfigError, match="relative to the clock") as exc:
        load_spec(_spec(steps=[{"kind": "window", "start": "now"}]))
    assert exc.value.field == "steps.0.start"


# -- JSON-able values ----------------------------------------------------------------


@pytest.mark.unit
def test_non_json_metadata_is_refused():
    with pytest.raises(SignalConfigError, match="JSON can't represent") as exc:
        load_spec(_spec(metadata={"created": pd.Timestamp("2025-01-01")}))
    assert exc.value.field == "metadata.created"
    with pytest.raises(ValidationError, match=r"metadata\.created"):
        SignalSpec(name="x", metadata={"created": pd.Timestamp("2025-01-01")})
    assert SignalSpec(name="x", metadata={"a": [1, 2.5, None, {"b": True}]})


@pytest.mark.unit
def test_load_spec_dict_with_a_numpy_integer_is_a_config_error():
    spec = _spec(steps=[{"kind": "hampel", "streams": ["flow"], "window": np.int64(5)}])
    with pytest.raises(SignalConfigError, match="JSON can't represent") as exc:
        load_spec(spec)
    assert exc.value.field == "steps.0.window"


@pytest.mark.unit
def test_a_source_cycle_is_a_config_error(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"$source": "b.json"}))
    (tmp_path / "b.json").write_text(json.dumps({"$source": "a.json"}))
    with pytest.raises(SignalConfigError, match="cycle"):
        load_spec(_spec(metadata={"$source": "a.json"}), base_dir=tmp_path)


@pytest.mark.unit
def test_the_same_source_twice_is_not_a_cycle(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps({"k": 1}))
    ref = {"$source": "t.json"}
    spec = load_spec(_spec(metadata={"a": ref, "b": ref}), base_dir=tmp_path)
    assert spec.metadata == {"a": {"k": 1}, "b": {"k": 1}}


@pytest.mark.unit
def test_a_non_string_source_is_a_config_error():
    with pytest.raises(SignalConfigError, match="must be the path of a JSON file"):
        load_spec(_spec(metadata={"$source": 5}))


# -- a spec passed where steps belong ------------------------------------------------


@pytest.mark.unit
def test_a_spec_passed_as_steps_says_how_to_use_it():
    spec = SignalSpec(name="inner")
    with pytest.raises(ValidationError, match="from_spec"):
        SignalSpec(name="x", steps=spec)
    with pytest.raises(ValidationError, match="from_spec"):
        fs.Pipeline(spec)
    with pytest.raises(ValidationError, match="not a step"):
        SignalSpec(name="x", steps=[fs.spec.CsvOutput(path="a.csv")])


# -- plugin entry points -------------------------------------------------------------


class _GoodPlugin(StepSpec):
    kind: Literal["review_fix_plugin"] = "review_fix_plugin"


class _EntryPoint:
    def __init__(self, name, obj):
        self.name = name
        self.value = f"pkg:{name}"
        self._obj = obj

    def load(self):
        if isinstance(self._obj, Exception):
            raise self._obj
        return self._obj


@pytest.mark.unit
def test_a_broken_entry_point_warns_and_the_others_still_load(monkeypatch):
    points = [
        _EntryPoint("broken", ImportError("plugin_pkg needs torch")),
        _EntryPoint("good", _GoodPlugin),
        _EntryPoint("module", types.ModuleType("registers_on_import")),
    ]
    monkeypatch.setattr(registry, "entry_points", lambda *, group: points)
    monkeypatch.setattr(registry, "_entry_points_loaded", False)
    try:
        with pytest.warns(SignalWarning, match="'broken'.*plugin_pkg needs torch"):
            load_entry_points()
        assert get_step_class("review_fix_plugin") is _GoodPlugin
        with pytest.raises(SignalConfigError, match="unknown step kind"):
            get_step_class("no_such_kind")
    finally:
        registry.unregister_step("review_fix_plugin")
