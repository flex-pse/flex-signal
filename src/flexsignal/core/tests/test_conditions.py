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


# -- M2: types, helpers and formulas ---------------------------------------------


def _frame():
    index = pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC")
    values = pd.DataFrame(
        {"C": [1.0, 0, 0, 0, 0, 1], "A": [1.0, 2, np.nan, 4, 5, 6]}, index=index
    )
    sf = fs.SignalFrame.from_frame(values)
    return fs.clean.Range("A", max=4.5).transform(sf, {})  # A[4:] OUT_OF_RANGE


@pytest.mark.unit
def test_ampersand_and_pipe_bind_like_and_or():
    # As in pandas.eval: 'a > 1 & b < 4' is '(a > 1) and (b < 4)'.
    assert evaluate_condition(_values(), "flow > 10 & pump_on == 1").tolist() == [
        False,
        True,
        False,
        True,
    ]


@pytest.mark.unit
def test_every_comparison_with_a_missing_value_is_false():
    assert evaluate_condition(_values(), "flow != 0").tolist() == [
        False,
        True,
        False,  # NaN
        True,
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("flow + 1", "gives numbers"),
        ("flow and pump_on == 1", "needs True/False"),
        ("(flow > 1) + 1 > 0", "needs numbers"),
        ("abs(flow > 1) > 0", "needs numbers"),
        ("abs > 0", "is a function"),
        ("arctan2(flow) > 0", "takes 2 argument"),
        ("held(flow > 0, '-5min')", "0 or more"),
        ("held(flow > 0)", "condition and a duration"),
        ("flagged(flow, 'NOPE')", "unknown flag"),
        ("flagged(flow)", "one or more flag names"),
        ("is_raw(flow + 1)", "stream name"),
        ("(flow > 1) == 1", "mixes"),
    ],
)
def test_condition_types_are_checked_on_load(expression, message):
    with pytest.raises(ValueError, match=message):
        check_condition(expression)


@pytest.mark.unit
def test_held_counts_from_the_start_of_each_unbroken_run():
    sf = _frame()
    # C == 0 from 00:10 to 00:40; it has held 20 minutes from 00:30.
    held = evaluate_condition(sf, 'held(C == 0, "20min")').tolist()
    assert held == [False, False, False, True, True, False]
    assert evaluate_condition(sf, "held(C == 0, '0min')").tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]


@pytest.mark.unit
def test_flag_helpers_read_the_flags():
    sf = _frame()
    assert evaluate_condition(sf, "valid(A)").tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert evaluate_condition(sf, "is_raw(A)").tolist() == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert evaluate_condition(sf, "flagged(A, 'OUT_OF_RANGE')").tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert evaluate_condition(sf, "flagged(A, 'MISSING', 'STUCK')").tolist()[2]
    with pytest.raises(SignalConfigError, match="needs? .*SignalFrame|SignalFrame"):
        evaluate_condition(sf.values, "is_raw(A)")


@pytest.mark.unit
def test_formulas():
    from flexsignal.core import check_formula, evaluate_formula
    from flexsignal.core.conditions import formula_names

    values = _values()
    out = evaluate_formula(values, "a + b * `flow rate`", {"a": 1.0, "b": 2.0})
    assert out.tolist() == [3.0, 5.0, 7.0, 9.0]
    assert np.isnan(evaluate_formula(values, "flow * 2")[2])
    assert formula_names("a + b * flow + maximum(pump_on, 0)", ["a", "b"]) == [
        "flow",
        "pump_on",
    ]
    with pytest.raises(ValueError, match="gives True/False"):
        check_formula("flow > 1")
    with pytest.raises(ValueError, match="only in conditions"):
        check_formula("held(flow > 0, '5min')")
    with pytest.raises(SignalConfigError, match="unknown stream 'zzz'"):
        evaluate_formula(values, "a * zzz", {"a": 1.0})
