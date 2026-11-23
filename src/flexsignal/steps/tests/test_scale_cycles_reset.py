"""Scale, the shared CycleTerm, RateOfChange.reset_after and Expression's EXCLUDED."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.testing import synthetic


def _hourly(values):
    t = pd.date_range("2025-01-01", periods=len(values), freq="h", tz="UTC")
    return fs.SignalFrame(
        pd.DataFrame({"x": values}, index=t), meta={"x": {"units": "kg/hr"}}
    )


def _run(step, sf):
    return fs.run(fs.Pipeline([step], name="t").spec, sf).signal


@pytest.mark.unit
def test_scale_multiplies_sets_units_and_keeps_flags():
    sf = _run(fs.clean.Range("x", max=5), _hourly([1.0, 2.0, 9.0, 4.0]))
    out = _run(fs.select.Scale("x", factor=0.001, units="m^3/hr"), sf)
    np.testing.assert_allclose(out.values["x"], [0.001, 0.002, np.nan, 0.004])
    assert out.units["x"] == "m^3/hr"
    assert out.flags.equals(sf.flags)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [0.0, float("inf"), float("nan")])
def test_scale_refuses_a_useless_factor(factor):
    with pytest.raises(ValidationError, match="finite, non-zero"):
        fs.select.Scale("x", factor=factor, units=None)


@pytest.mark.unit
def test_cycle_terms_keep_each_steps_default_harmonics():
    term = fs.steps.impute.CycleTerm(period="1D")
    assert term.harmonics is None and term.pairs(2) == 2 and term.pairs(3) == 3
    assert fs.steps.impute.CycleTerm(period="1D", harmonics=5).pairs(3) == 5


@pytest.mark.component
def test_regression_without_harmonics_still_uses_three_pairs():
    data = synthetic.plant(seed=1, periods=3 * 288).with_gaps("flow", count=3).signal()

    def filled(fourier):
        step = fs.impute.Regression("flow", features=["pressure"], fourier=fourier)
        return fs.run(fs.Pipeline([step], name="r").spec, data).signal.values

    assert filled([{"period": "1D"}]).equals(filled([{"period": "1D", "harmonics": 3}]))


@pytest.mark.unit
def test_reset_after_flags_a_bad_first_value_instead_of_the_good_data():
    values = 10 + 0.5 * np.sin(np.arange(48.0))
    values[0] = 1000.0
    without = _run(fs.clean.RateOfChange("x", max_rate=5), _hourly(values))
    with_reset = _run(
        fs.clean.RateOfChange("x", max_rate=5, reset_after=3), _hourly(values)
    )
    assert without.has_flag(fs.Flag.RATE_OF_CHANGE)["x"].sum() == 47  # unchanged
    flagged = with_reset.has_flag(fs.Flag.RATE_OF_CHANGE)["x"]
    assert list(np.flatnonzero(flagged)) == [0]


@pytest.mark.unit
def test_reset_after_keeps_a_real_step_but_still_catches_a_spike():
    values = np.full(48, 10.0)
    values[20:] = 50.0  # a genuine step, faster than max_rate
    values[10] = 90.0  # a one-sample spike
    out = _run(fs.clean.RateOfChange("x", max_rate=5, reset_after=3), _hourly(values))
    assert list(np.flatnonzero(out.has_flag(fs.Flag.RATE_OF_CHANGE)["x"])) == [10]


@pytest.mark.unit
def test_reset_after_needs_a_run_of_agreeing_values():
    values = np.full(12, 10.0)
    values[[3, 5, 7]] = [80.0, 200.0, 80.0]  # rejected, but not a level
    out = _run(fs.clean.RateOfChange("x", max_rate=5, reset_after=2), _hourly(values))
    assert list(np.flatnonzero(out.has_flag(fs.Flag.RATE_OF_CHANGE)["x"])) == [3, 5, 7]


@pytest.mark.unit
def test_an_expression_can_exclude_values():
    sf = _hourly([1.0, -1.0, 2.0])
    out = _run(fs.clean.Expression("x", invalid_when="x < 0", flag="EXCLUDED"), sf)
    assert out.has_flag(fs.Flag.EXCLUDED)["x"].tolist() == [False, True, False]
