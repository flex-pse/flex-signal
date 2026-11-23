"""fit_data's window, rename and time zone; GridValues.resampled; strict overrides."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError
from flexsignal.handoff import FitPrep, GridOptions, fit_data, to_grid


def _frames():
    t = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
    feed = pd.DataFrame({"feed": np.arange(48.0)}, index=t)
    gas = pd.DataFrame({"gas": 2 * np.arange(48.0)}, index=t)
    return (feed, {"feed": "kg/hr"}), (gas, {"gas": "m^3/hr"})


@pytest.mark.unit
def test_a_window_keeps_rows_from_start_to_before_end():
    X, y = _frames()
    fit = fit_data(X, y, FitPrep(start="2025-01-01T06:00Z", end="2025-01-02T00:00"))
    assert fit.X.index[0] == pd.Timestamp("2025-01-01T06:00Z")
    assert fit.X.index[-1] == pd.Timestamp("2025-01-01T23:00Z")
    assert len(fit.X) == 18
    assert {"step": "window", "start": "2025-01-01T06:00Z"}.items() <= fit.steps[
        1
    ].items()


@pytest.mark.unit
def test_rename_gives_the_regressor_its_names_and_units():
    X, y = _frames()
    fit = fit_data(X, y, {"rename": {"feed": "flow_in_feed", "gas": "biogas_volume"}})
    assert list(fit.X.columns) == ["flow_in_feed"]
    assert list(fit.y.columns) == ["biogas_volume"]
    assert fit.input_units == {"flow_in_feed": "kg/hr"}
    assert fit.output_units == "m^3/hr"


@pytest.mark.unit
def test_rename_refuses_unknown_columns_and_clashes():
    X, y = _frames()
    with pytest.raises(SignalConfigError, match="aren't in the data"):
        fit_data(X, y, {"rename": {"nope": "x"}})
    with pytest.raises(SignalConfigError, match="several columns the name"):
        fit_data(X, y, {"rename": {"feed": "x", "gas": "x"}})


@pytest.mark.unit
def test_timezone_gives_the_models_naive_wall_clock():
    X, y = _frames()
    fit = fit_data(X, y, {"timezone": "America/Los_Angeles"})
    assert fit.X.index.tz is None
    assert fit.X.index[0] == pd.Timestamp("2024-12-31T16:00")  # UTC-8
    assert fit_data(X, y).X.index.tz is not None  # default unchanged


@pytest.mark.unit
@pytest.mark.parametrize(
    "prep",
    [
        {"start": "now"},
        {"end": "NaT"},
        {"timezone": "Mars/Olympus"},
        {"start": "2025-01-02", "end": "2025-01-01"},
    ],
)
def test_bad_window_or_zone_is_refused(prep):
    with pytest.raises(ValueError):
        FitPrep(**prep)


@pytest.mark.unit
def test_upsampled_steps_are_counted_as_resampled_not_estimated():
    t = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    hourly = fs.SignalFrame(pd.DataFrame({"feed": np.arange(6.0)}, index=t))
    fine = fs.run(
        fs.Pipeline([fs.select.Align(freq="15min", tolerance="1h")], name="up").spec,
        hourly,
    )
    grid = pd.date_range("2025-01-01", periods=20, freq="15min")
    values = to_grid(fine.signal, grid, GridOptions(gaps="error"))
    assert values.estimated == {"feed": 0}
    assert values.resampled == {"feed": 15}  # 3 of every 4 steps interpolated


@pytest.mark.unit
def test_strict_overrides_need_a_path_for_every_column():
    t = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame({"feed": [1.0] * 4, "biogas": [2.0] * 4}, index=t)
    values = to_grid(data, pd.date_range("2025-01-01", periods=4, freq="h"))
    assert set(values.to_overrides({"feed": "plant.feed.flow"})) == {
        "plant.feed.flow",
        "biogas",
    }
    with pytest.raises(SignalConfigError, match=r"\['biogas'\] have no model path"):
        values.to_overrides({"feed": "plant.feed.flow"}, strict=True)
