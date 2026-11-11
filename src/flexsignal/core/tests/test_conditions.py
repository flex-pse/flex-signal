"""Conditions can read streams but never run code."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.core import SignalConfigError, check_condition, evaluate_condition


def _values():
    return pd.DataFrame(
        {
            "flow": [0.0, 50.0, np.nan, 120.0],
            "pump_on": [0.0, 1.0, 1.0, 1.0],
            "flow rate": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("pump_on == 1 and flow > 10", [False, True, False, True]),
        ("(pump_on == 1) & ~(flow > 100)", [False, True, True, False]),
        ("not flow > 10 or pump_on == 0", [True, False, True, False]),
        ("abs(flow - 100) < 30", [False, False, False, True]),
        ("sqrt(flow) ** 2 >= 50", [False, True, False, True]),
        ("`flow rate` * 2 > 5", [False, False, True, True]),
        ("0 < flow <= 50", [False, True, False, False]),  # chained comparison
        ("flow > -1e3", [True, True, False, True]),  # NaN compares False
        ("arctan2(flow, pump_on + 1) > 1.5", [False, True, False, True]),
    ],
)
def test_allowed_conditions(expression, expected):
    assert evaluate_condition(_values(), expression).tolist() == expected


# Each tries to reach Python internals or run code. None may get past the check,
# whichever way it is written. The payloads only read harmless values.
ESCAPES = [
    "flow.__class__.__name__ == 'Series'",
    "flow.__class__.__init__.__globals__['__name__'] == 'x'",
    "flow.abs() > 0",
    "flow.to_csv('x.csv') == 0",
    "__import__('os').getcwd() == ''",
    "open('x') == 0",
    "eval('1') == 1",
    "getattr(flow, 'abs')() > 0",
    "(lambda: 1)() == 1",
    "[x for x in flow] == 0",
    "{1: 2}[1] == 2",
    "flow if pump_on else 0",
    "abs(flow, key=1) > 0",
    "abs(*flow) > 0",
    "flow > @limit",
    "'a' == 'a'",
    "flow in [1, 2]",
    "9 ** 9 ** 9 > 0",
    "flow ** pump_on > 0",
    "(x := 1) == 1",
    "flow > 0; import os",
]


@pytest.mark.unit
@pytest.mark.parametrize("expression", ESCAPES)
def test_conditions_cannot_reach_python_internals(expression):
    with pytest.raises(ValueError):
        check_condition(expression)
    with pytest.raises(SignalConfigError) as info:
        evaluate_condition(_values(), expression, field="keep_when")
    assert info.value.field == "keep_when"


@pytest.mark.unit
def test_backticks_cannot_smuggle_code():
    # Text inside backticks is only ever a stream name, never code.
    with pytest.raises(SignalConfigError):
        evaluate_condition(_values(), "`flow.__class__` > 0")


@pytest.mark.unit
def test_overlong_conditions_are_refused():
    with pytest.raises(ValueError, match="limit"):
        check_condition(" or ".join(["flow > 0"] * 200))


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["expression", "mask"])
def test_a_spec_with_an_unsafe_condition_fails_on_load(kind):
    step = {"kind": kind, "streams": ["flow"]}
    key = "invalid_when" if kind == "expression" else "keep_when"
    step[key] = "flow.__class__.__init__.__globals__['__name__'] == 'x'"
    spec = {"schema_version": fs.SCHEMA_VERSION, "steps": [step]}
    with pytest.raises(SignalConfigError, match="not allowed") as info:
        fs.load_spec(spec)
    assert info.value.field == f"steps.0.{key}"
    with pytest.raises(ValidationError, match="can't be called"):  # built in Python
        fs.clean.Expression("flow", invalid_when="flow.abs() > 0")
