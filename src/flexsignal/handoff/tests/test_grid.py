"""to_grid: data lined up with a model's time steps.

Most of these are flex-orchestrator's model_inputs tests, ported as pure-data
tests with the same data and expected values,
so behaviour can't drift when flex-orchestrator switches to these functions.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError, SignalDataError
from flexsignal.handoff import GridOptions, data_window, to_grid, values_hash

HOURS = pd.date_range("2025-07-07", periods=72, freq="h", tz="UTC")
GRID = pd.date_range("2025-07-08", periods=8, freq="15min")  # 00:00 to 01:45


def ops():
    """Hourly demand 0, 1, 2, ... (m^3/hr) and a unitless count, as stored."""
    frame = pd.DataFrame(
        {"demand": np.arange(72.0), "count": np.arange(72.0) * 10}, HOURS
    )
    return frame, {"demand": "m^3/hr"}


def demand(grid=GRID, **options):
    return to_grid(ops(), grid, GridOptions(**options), columns=["demand"])


@pytest.mark.unit
def test_hourly_data_fills_every_step():
    found = demand()
    # 24 at 00:00 and 25 at 01:00 (UTC); the quarter hours between are filled.
    assert found.values["demand"] == [
        24.0,
        24.25,
        24.5,
        24.75,
        25.0,
        25.25,
        25.5,
        25.75,
    ]
    assert found.units == {"demand": "m^3/hr"}
    assert found.filled == {"demand": 6}
    assert found.window == {
        "start": "2025-07-08 00:00:00",
        "end": "2025-07-08 02:00:00",
    }
    overrides = found.to_overrides(rename={"demand": "plant.d"})
    assert overrides == {
        "plant.d": {"value": found.values["demand"], "units": "m^3/hr"}
    }
    # The hash flex-orchestrator records: sha256 of sorted, compact JSON.
    text = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
    expected = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert (
        found.data_hash(rename={"demand": "plant.d"})
        == expected
        == values_hash(overrides)
    )


@pytest.mark.unit
def test_a_column_without_a_unit_is_a_plain_list():
    found = to_grid(ops(), GRID, {"gaps": "ffill"}, columns=["count"])
    assert found.values["count"] == [240.0] * 4 + [250.0] * 4
    assert found.to_overrides() == {"count": [240.0] * 4 + [250.0] * 4}


@pytest.mark.unit
def test_the_facility_time_zone():
    # 00:00 in Los Angeles (PDT) is 07:00 UTC, when the demand was 31.
    assert demand(timezone="America/Los_Angeles").values["demand"][0] == 31.0


@pytest.mark.unit
def test_daylight_saving_days():
    stamps = pd.date_range("2025-03-08", "2025-11-04", freq="h", tz="UTC")
    data = pd.DataFrame({"v": np.arange(len(stamps), dtype=float)}, stamps)
    zone = {"timezone": "America/Los_Angeles"}
    spring = pd.date_range("2025-03-09", periods=24, freq="h")
    found = to_grid(data, spring, zone)
    values = found.values["v"]
    # 02:00 doesn't exist that day: interpolated between 01:00 and 03:00.
    assert values[2] == (values[1] + values[3]) / 2
    assert found.filled == {"v": 1}
    autumn = pd.date_range("2025-11-02", periods=24, freq="h")
    values = to_grid(data, autumn, zone).values["v"]
    # 01:00 happens twice; its two samples average into one step.
    assert values[1] == values[0] + 1.5
    assert values[2] == values[0] + 3


@pytest.mark.unit
def test_resample_aggregates_onto_a_coarser_step():
    grid = pd.date_range("2025-07-08", periods=2, freq="4h")
    assert demand(grid).values["demand"] == [25.5, 29.5]
    assert demand(grid, resample={"agg": "sum"}).values["demand"] == [102.0, 118.0]


@pytest.mark.unit
def test_a_backward_step_takes_the_interval_ending_at_it():
    grid = pd.date_range("2025-07-08", periods=2, freq="4h")
    backward = {"agg": "mean", "window": "backward"}
    found = demand(grid, resample=backward)
    # 00:00 takes 21:00 to 00:00 (21, 22, 23, 24); forward took 24 to 27.
    assert found.values["demand"] == [22.5, 26.5]
    assert found.window == {
        "start": "2025-07-07 20:00:00",
        "end": "2025-07-08 04:00:00",
    }
    total = demand(grid, resample={**backward, "agg": "sum"})
    assert total.values["demand"] == [90.0, 106.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("window", "expected"),
    [
        # The sample at 00:00 (24) covers 00:00 to 01:00: a quarter on each step.
        ("forward", [6.0] * 4 + [6.25] * 4),
        # The sample at 01:00 (25) covers 00:00 to 01:00, ending at it.
        ("backward", [6.0] + [6.25] * 4 + [6.5] * 3),
    ],
)
def test_a_sum_of_coarser_data_is_split_over_the_steps(window, expected):
    found = demand(resample={"agg": "sum", "window": window})
    assert found.values["demand"] == expected
    assert found.filled == {"demand": 0}


@pytest.mark.unit
def test_a_sum_splits_only_over_whole_steps():
    grid = pd.date_range("2025-07-08", periods=3, freq="40min")
    with pytest.raises(SignalDataError, match="whole number of steps"):
        demand(grid, resample={"agg": "sum"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        ("interpolate", [24.0, 24.5, 25.0]),
        ("ffill", [24.0, 24.0, 25.0]),
        ("zero", [24.0, 0.0, 25.0]),
    ],
)
def test_gap_rules(gaps, expected):
    grid = pd.date_range("2025-07-08", periods=3, freq="30min")
    assert demand(grid, gaps=gaps).values["demand"] == expected


@pytest.mark.unit
def test_gaps_that_cant_be_filled():
    with pytest.raises(SignalDataError, match="no value at 6 of the model's 8"):
        demand(gaps="error")
    with pytest.raises(SignalDataError, match="with limit 2"):
        demand(limit=2)
    after = pd.date_range("2025-07-09T23:00", periods=4, freq="30min")
    with pytest.raises(SignalDataError, match="no value at 3 of the model's 4"):
        demand(after)  # 23:00 is the last sample


@pytest.mark.unit
def test_reduce_gives_one_number():
    last = demand(reduce="last")
    assert last.values["demand"] == 25.0
    assert last.to_overrides() == {"demand": {"value": 25.0, "units": "m^3/hr"}}
    assert demand(reduce="mean").values["demand"] == 24.5
    everything = to_grid(ops(), None, {"reduce": "max"}, columns=["demand"])
    assert everything.values["demand"] == 71.0
    assert everything.window is None


@pytest.mark.unit
def test_a_model_without_time_steps_takes_only_reduced_inputs():
    with pytest.raises(SignalConfigError, match="no time axis"):
        to_grid(ops(), None)


@pytest.mark.unit
def test_offset_reads_another_window():
    found = demand(offset="-1D", reduce="first")
    assert found.values["demand"] == 0.0
    assert found.window["start"] == "2025-07-07 00:00:00"


@pytest.mark.unit
def test_data_years_read_another_years_dates():
    later = pd.date_range("2030-07-08", periods=2, freq="h")
    found = demand(later, data_years={2030: 2025})
    assert found.values["demand"] == [24.0, 25.0]
    assert found.window["start"] == "2025-07-08 00:00:00"
    # Then offset shifts within the data year.
    shifted = demand(later, data_years={2030: 2025}, offset="-1D")
    assert shifted.values["demand"] == [0.0, 1.0]
    # A year that isn't listed reads its own dates, which have no data.
    with pytest.raises(SignalDataError, match="no samples in the window"):
        demand(later, data_years={2035: 2025})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"timezone": "Mars/Olympus"}, "unknown time zone"),
        ({"offset": "a while"}, "isn't a duration"),
        ({"reduce": "last", "gaps": "zero"}, "don't apply"),
        ({"limit": 0}, "greater than or equal to 1"),
        ({"gaps": "guess"}, "Input should be"),
        ({"resample": {"window": "sideways"}}, "Input should be"),
    ],
)
def test_invalid_options(fields, message):
    with pytest.raises(ValidationError, match=message):
        GridOptions(**fields)


# -- beyond flex-orchestrator ------------------------------------------------------


@pytest.mark.unit
def test_data_window_is_the_horizon_plus_a_margin_in_utc():
    start, end = data_window(GRID, GridOptions(timezone="America/Los_Angeles"))
    # 8 July 00:00 to 02:00 PDT, a day either side, in UTC.
    assert start == pd.Timestamp("2025-07-07 07:00", tz="UTC")
    assert end == pd.Timestamp("2025-07-09 09:00", tz="UTC")
    start, end = data_window(GRID, GridOptions(reduce="first"))
    assert (start, end) == (
        pd.Timestamp("2025-07-08 00:00", tz="UTC"),
        pd.Timestamp("2025-07-08 02:00", tz="UTC"),
    )
    assert data_window(None) is None


@pytest.mark.unit
def test_a_signal_frame_brings_units_and_reports_estimated_steps():
    frame, units = ops()
    frame.loc["2025-07-08 01:00", "demand"] = np.nan  # a gap...
    sf = fs.SignalFrame.from_frame(frame, units=units)
    filled = fs.impute.Interpolate("demand", max_gap="3h").transform(sf, {})
    found = to_grid(filled, GRID, columns=["demand"])
    assert found.units == {"demand": "m^3/hr"}
    # ...filled by interpolate: the 01:00 step rests on an imputed sample.
    assert found.estimated == {"demand": 1}
    assert found.values["demand"][4] == 25.0
    plain = to_grid(ops(), GRID, columns=["demand"])
    assert plain.estimated == {"demand": 0}


@pytest.mark.unit
def test_inputs_are_checked():
    with pytest.raises(SignalConfigError, match="unknown column"):
        to_grid(ops(), GRID, columns=["nope"])
    with pytest.raises(SignalConfigError, match="naive local time"):
        to_grid(ops(), GRID.tz_localize("UTC"))
    with pytest.raises(SignalConfigError, match="expected a SignalFrame"):
        to_grid([1, 2, 3], GRID)
    assert Flag.CORRECTED  # corrected samples count as estimated too (see _ESTIMATED)
