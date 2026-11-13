import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError, SignalDataError, flag_names


def _sf(data, start="2025-01-01", freq="15min", units=None):
    index = pd.date_range(
        start, periods=len(next(iter(data.values()))), freq=freq, tz="UTC"
    )
    return fs.SignalFrame.from_frame(pd.DataFrame(data, index=index), units=units)


def _run(step, sf):
    return step.transform(sf, step.fit(sf))


@pytest.mark.unit
def test_align_nearest_keeps_exact_samples_and_flags_moved_ones():
    times = pd.to_datetime(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:16",
            "2025-01-01 00:44",
            "2025-01-01 01:30",
        ],
        utc=True,
    )
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=times)
    )
    out = _run(fs.select.Align(freq="15min"), sf)
    assert list(out.index.strftime("%H:%M")) == [
        "00:00",
        "00:15",
        "00:30",
        "00:45",
        "01:00",
        "01:15",
        "01:30",
    ]
    # 00:15 takes 00:16 and 00:45 takes 00:44 (1 min away). 00:30 is 14 min from
    # both, so either is nearest. 01:00 has no sample within the 15 min tolerance.
    assert out.values["x"].iloc[1] == 2.0 and out.values["x"].iloc[3] == 3.0
    assert out.values["x"].iloc[2] in (2.0, 3.0)
    flags = [flag_names(int(f)) for f in out.flags["x"]]
    assert flags[0] == [] and flags[-1] == []
    assert flags[1] == ["RESAMPLED"]
    assert flags[4] == ["MISSING"] and np.isnan(out.values["x"].iloc[4])


@pytest.mark.unit
def test_align_linear_interpolates_between_neighbours():
    times = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:20"], utc=True)
    sf = fs.SignalFrame.from_frame(pd.DataFrame({"x": [0.0, 20.0]}, index=times))
    out = _run(fs.select.Align(freq="5min", method="linear", tolerance="20min"), sf)
    assert out.values["x"].tolist() == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert flag_names(int(out.flags["x"].iloc[2])) == ["RESAMPLED"]
    # The samples are 20 min apart, more than the default tolerance of 2 x 5 min.
    default = _run(fs.select.Align(freq="5min", method="linear"), sf)
    assert default.values["x"].isna().tolist() == [False, True, True, True, False]


@pytest.mark.unit
def test_align_inner_and_outer_spans():
    a = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2025-01-01 00:00", periods=3, freq="1h", tz="UTC"),
    )
    b = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2025-01-01 01:00", periods=3, freq="1h", tz="UTC"),
    )
    sf = fs.SignalFrame.from_frame(pd.concat({"a": a, "b": b}, axis=1))
    assert len(_run(fs.select.Align(freq="1h", how="outer"), sf)) == 4
    assert len(_run(fs.select.Align(freq="1h", how="inner"), sf)) == 2


@pytest.mark.unit
def test_resample_aggregates_and_ors_flags():
    sf = _sf({"x": [1.0, np.nan, 3.0, 5.0, np.nan, np.nan, 7.0, 9.0]})
    out = _run(fs.select.Resample(freq="30min"), sf)
    assert out.values["x"].tolist()[:2] == [1.0, 4.0]
    assert np.isnan(out.values["x"].iloc[2])
    assert flag_names(int(out.flags["x"].iloc[0])) == ["MISSING", "RESAMPLED"]
    assert flag_names(int(out.flags["x"].iloc[1])) == ["RESAMPLED"]
    assert flag_names(int(out.flags["x"].iloc[2])) == ["MISSING"]
    strict = _run(fs.select.Resample(freq="30min", min_count=2), sf)
    assert np.isnan(strict.values["x"].iloc[0])


@pytest.mark.unit
def test_window_and_mask():
    sf = _sf({"x": [1.0, 2, 3, 4], "on": [1, 0, 1, 1]})
    assert (
        len(
            _run(
                fs.select.Window(start="2025-01-01T00:15:00", end="2025-01-01T00:45"),
                sf,
            )
        )
        == 2
    )
    assert len(_run(fs.select.Window(last="30min"), sf)) == 2
    masked = _run(fs.select.Mask(keep_when="on == 1", streams="x"), sf)
    assert np.isnan(masked.values["x"].iloc[1])
    assert flag_names(int(masked.flags["x"].iloc[1])) == ["EXCLUDED"]
    assert masked.values["on"].iloc[1] == 0
    assert len(_run(fs.select.Mask(keep_when="on == 1", drop_rows=True), sf)) == 3
    with pytest.raises(ValueError, match="start, end or last"):
        fs.select.Window()


@pytest.mark.unit
def test_rename_moves_metadata():
    sf = _sf({"FT_101": [1.0, 2.0]}, units={"FT_101": "m^3/hr"})
    out = _run(fs.select.Rename(mapping={"FT_101": "flow"}), sf)
    assert out.columns == ["flow"] and out.units == {"flow": "m^3/hr"}
    with pytest.raises(SignalConfigError, match="unknown stream"):
        _run(fs.select.Rename(mapping={"nope": "x"}), sf)


@pytest.mark.unit
def test_combine_coalesce_and_mean():
    sf = _sf(
        {"a": [1.0, np.nan, np.nan], "b": [10.0, 20.0, np.nan]},
        units={"a": "bar", "b": "bar"},
    )
    out = _run(fs.select.Combine(output="c", streams=["a", "b"]), sf)
    assert out.values["c"].tolist()[:2] == [1.0, 20.0] and np.isnan(
        out.values["c"].iloc[2]
    )
    assert out.units["c"] == "bar"
    assert flag_names(int(out.flags["c"].iloc[0])) == ["DERIVED"]
    assert "MISSING" in flag_names(int(out.flags["c"].iloc[2]))
    mean = _run(fs.select.Combine(output="m", streams=["a", "b"], how="mean"), sf)
    assert mean.values["m"].iloc[0] == 5.5
    mixed = _sf({"a": [1.0], "b": [2.0]}, units={"a": "bar", "b": "kPa"})
    with pytest.raises(SignalConfigError, match="different units"):
        _run(fs.select.Combine(output="c", streams=["a", "b"]), mixed)


@pytest.mark.unit
def test_shift_needs_regular_grid_and_whole_steps():
    sf = _sf({"x": [1.0, 2.0, 3.0, 4.0]})
    out = _run(fs.select.Shift("x", by="15min"), sf)
    assert np.isnan(out.values["x"].iloc[0]) and out.values["x"].iloc[1] == 1.0
    assert out.flags["x"].iloc[0] == Flag.MISSING
    back = _run(fs.select.Shift("x", by="-30min"), sf)
    assert back.values["x"].iloc[0] == 3.0
    with pytest.raises(SignalConfigError, match="whole number"):
        _run(fs.select.Shift("x", by="10min"), sf)
    irregular = fs.SignalFrame.from_frame(
        pd.DataFrame(
            {"x": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:20"], utc=True
            ),
        )
    )
    with pytest.raises(SignalDataError, match="regular"):
        _run(fs.select.Shift("x", by="15min"), irregular)
