"""The step registry in flexsignal.core, with step classes built on StepSpec.

These live with the spec tests because StepSpec is in the spec layer, which
sits above core: tests inside core may not import it (see .importlinter).
"""

import types
from typing import Literal

import pytest

from flexsignal.core import SignalConfigError, registry
from flexsignal.core.registry import (
    STEP_ENTRY_POINT_GROUP,
    find_step,
    get_step_class,
    load_entry_points,
    register_step,
    unregister_step,
)
from flexsignal.spec import StepSpec


class _NoKindDefault(StepSpec):
    pass


class _First(StepSpec):
    kind: Literal["registry_test_dupe"] = "registry_test_dupe"


class _Second(StepSpec):
    kind: Literal["registry_test_dupe"] = "registry_test_dupe"


class _FromPlugin(StepSpec):
    kind: Literal["registry_test_plugin"] = "registry_test_plugin"


@pytest.fixture
def forget():
    """Unregister the step kinds a test registers."""
    kinds = []
    yield kinds
    for kind in kinds:
        unregister_step(kind)


@pytest.mark.unit
def test_a_step_whose_kind_has_no_string_default_is_refused():
    with pytest.raises(SignalConfigError, match="_NoKindDefault can't be registered"):
        register_step(_NoKindDefault)
    with pytest.raises(SignalConfigError, match="needs a string default"):
        register_step(object)


@pytest.mark.unit
def test_a_kind_taken_by_another_class_needs_replace(forget):
    forget.append("registry_test_dupe")
    assert register_step(_First) is _First
    assert register_step(_First) is _First  # the same class again is fine
    with pytest.raises(SignalConfigError, match="already registered by") as exc:
        register_step(_Second)
    assert "_First" in exc.value.message
    assert find_step("registry_test_dupe") is _First
    assert register_step(_Second, replace=True) is _Second
    assert get_step_class("registry_test_dupe") is _Second


class _EntryPoint:
    def __init__(self, obj):
        self._obj = obj

    def load(self):
        return self._obj


@pytest.mark.unit
def test_entry_points_register_classes_and_import_modules(monkeypatch, forget):
    groups = []

    def fake_entry_points(*, group):
        groups.append(group)
        module = types.ModuleType("registers_on_import")
        return [_EntryPoint(_FromPlugin), _EntryPoint(module)]

    forget.append("registry_test_plugin")
    monkeypatch.setattr(registry, "entry_points", fake_entry_points)
    monkeypatch.setattr(registry, "_entry_points_loaded", False)
    assert find_step("registry_test_plugin") is None
    load_entry_points()
    assert groups == [STEP_ENTRY_POINT_GROUP]
    assert get_step_class("registry_test_plugin") is _FromPlugin
    load_entry_points()  # only once
    assert groups == [STEP_ENTRY_POINT_GROUP]
