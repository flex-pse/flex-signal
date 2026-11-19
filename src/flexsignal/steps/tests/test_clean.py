"""M1 acceptance: each cleaner flags exactly the faults injected into synthetic data."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError, SignalDataError
from flexsignal.testing import synthetic


def _flagged(sf, flag, stream="flow"):
    return ((sf.flags[stream] & int(flag)) != 0).to_numpy()


def _run(step, sf):
    return step.transform(sf, step.fit(sf))


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_range_flags_exactly_the_sentinels(seed):
    case = synthetic.plant(seed=seed).with_out_of_range("flow", count=6, value=-999)
    out = _run(fs.clean.Range("flow", min=0, max=500), case.signal())
    expected = case.fault_mask("out_of_range", "flow").to_numpy()
    assert np.array_equal(_flagged(out, Flag.OUT_OF_RANGE), expected)
    assert out.values["flow"][expected].isna().all()
    untouched = ~expected
    assert np.array_equal(out.values["flow"][untouched], case.clean["flow"][untouched])


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rate_of_change_flags_exactly_the_spikes(seed):
    case = synthetic.plant(seed=seed).with_spikes("flow", count=8, size=40)
    out = _run(fs.clean.RateOfChange("flow", max_rate=200, per="1h"), case.signal())
    assert np.array_equal(
        _flagged(out, Flag.RATE_OF_CHANGE), case.fault_mask("spike", "flow").to_numpy()
    )


@pytest.mark.unit
def test_rate_of_change_accepts_a_genuine_step_once_time_allows():
    index = pd.date_range("2025-01-01", periods=6, freq="1min", tz="UTC")
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame({"x": [0, 0, 10, 10, 10, 10.0]}, index=index)
    )
    out = _run(fs.clean.RateOfChange("x", max_rate=4, per="1min"), sf)
    # 10 units at 4/min needs 2.5 min: rejected at +1 and +2 min, accepted at +3 min.
    assert _flagged(out, Flag.RATE_OF_CHANGE, "x").tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stuck_flags_exactly_the_flat_runs(seed):
    case = synthetic.plant(seed=seed).with_stuck("flow", count=2, length=36)
    out = _run(fs.clean.Stuck("flow", min_duration="1h"), case.signal())
    assert np.array_equal(
        _flagged(out, Flag.STUCK), case.fault_mask("stuck", "flow").to_numpy()
    )


@pytest.mark.unit
def test_stuck_ignores_short_runs_and_excluded_values():
    index = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    x = [1, 2, 2, 3, 0, 0, 0, 0, 0, 0.0]
    sf = fs.SignalFrame.from_frame(pd.DataFrame({"x": x}, index=index))
    out = _run(fs.clean.Stuck("x", min_duration="1h"), sf)
    assert _flagged(out, Flag.STUCK, "x").tolist() == [False] * 4 + [True] * 6
    out = _run(fs.clean.Stuck("x", min_duration="1h", exclude_values=[0]), sf)
    assert not _flagged(out, Flag.STUCK, "x").any()


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hampel_flags_exactly_the_spikes(seed):
    case = synthetic.plant(seed=seed).with_spikes("flow", count=8, size=40)
    step = fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=case.noise["flow"])
    out = _run(step, case.signal())
    assert np.array_equal(
        _flagged(out, Flag.OUTLIER), case.fault_mask("spike", "flow").to_numpy()
    )


@pytest.mark.unit
def test_hampel_min_scale_stops_false_alarms_on_flat_stretches():
    # Without a floor, a short window on a flat stretch can under-estimate the
    # noise and flag ordinary samples (seed 0 has one at 2025-01-08 15:40).
    case = synthetic.plant(seed=0)
    bare = _run(fs.clean.Hampel("flow", window="1h", n_sigma=6), case.signal())
    floored = _run(
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=case.noise["flow"]),
        case.signal(),
    )
    assert _flagged(bare, Flag.OUTLIER).sum() >= 1
    assert _flagged(floored, Flag.OUTLIER).sum() == 0


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tukey_learns_fences_and_flags_exactly_the_outliers(seed):
    case = synthetic.plant(seed=seed).with_out_of_range("flow", count=6, value=1000.0)
    step = fs.clean.Tukey("flow", k=3)
    state = step.fit(case.signal())
    lo, hi = state["fences"]["flow"]
    assert lo < case.clean["flow"].min() and hi > case.clean["flow"].max()
    out = step.transform(case.signal(), state)
    assert np.array_equal(
        _flagged(out, Flag.OUTLIER), case.fault_mask("out_of_range", "flow").to_numpy()
    )


@pytest.mark.unit
def test_expression_flags_exactly_where_the_condition_holds():
    case = synthetic.plant(seed=0).with_out_of_range("pressure", count=4, value=-5.0)
    out = _run(
        fs.clean.Expression("pressure", invalid_when="pressure < 0"), case.signal()
    )
    assert np.array_equal(
        _flagged(out, Flag.MANUAL, "pressure"),
        case.fault_mask("out_of_range", "pressure").to_numpy(),
    )


@pytest.mark.unit
def test_cleaners_leave_missing_values_and_other_streams_alone():
    case = synthetic.plant(seed=0).with_gaps("flow", count=3, length=2)
    out = _run(fs.clean.Range("flow", min=1e9), case.signal())
    gaps = case.fault_mask("gap", "flow").to_numpy()
    flags = out.flags["flow"].to_numpy()
    assert np.all(flags[gaps] == int(Flag.MISSING))
    assert np.all(flags[~gaps] == int(Flag.OUT_OF_RANGE))
    assert out.values["pressure"].equals(case.signal().values["pressure"])


@pytest.mark.unit
def test_range_quantity_bounds_are_converted_to_the_streams_units():
    pytest.importorskip("pint")
    sf = synthetic.plant().signal()
    same = _run(fs.clean.Range("flow", max={"value": 50, "units": "m^3/hr"}), sf)
    assert _flagged(same, Flag.OUT_OF_RANGE).any()
    # 25 L/s is 90 m^3/hr.
    in_litres = _run(fs.clean.Range("flow", max={"value": 25, "units": "L/s"}), sf)
    in_hours = _run(fs.clean.Range("flow", max=90.0), sf)
    pd.testing.assert_frame_equal(in_litres.flags, in_hours.flags)
    with pytest.raises(SignalConfigError, match="can't convert"):
        _run(fs.clean.Range("flow", max={"value": 50, "units": "kW"}), sf)


@pytest.mark.unit
def test_unknown_stream_and_too_little_data():
    sf = synthetic.plant().signal()
    with pytest.raises(SignalConfigError, match="unknown stream") as exc:
        _run(fs.clean.Hampel("nope", window="1h"), sf)
    assert exc.value.field == "streams"
    with pytest.raises(SignalDataError, match="at least 4"):
        fs.clean.Tukey("reference_flow").fit(
            sf.select(["reference_flow"]).evolve(
                values=sf.values[["reference_flow"]].iloc[:3]
            )
        )


# -- M2: filter and detotalize -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"method": "savgol", "window": "1h"},
        {"method": "butterworth", "cutoff": "2h"},
        {"method": "median", "window": "30min"},
    ],
)
def test_filter_reduces_noise_and_flags_what_it_smoothed(options):
    case = synthetic.plant(seed=4).with_gaps("flow", count=3, length=4)
    sf = case.signal()
    out = fs.clean.Filter("flow", **options).transform(sf, {})
    noise = sf.values["flow"] - case.underlying["flow"]
    left = out.values["flow"] - case.underlying["flow"]
    assert left.std() < 0.6 * noise.std()
    flags = out.flags["flow"].to_numpy()
    valid = sf.values["flow"].notna().to_numpy()
    assert np.all((flags[valid] & int(Flag.FILTERED)) != 0)
    assert out.values["flow"][~valid].isna().all()


@pytest.mark.unit
def test_filter_needs_its_options_and_a_sensible_cutoff():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="needs a window"):
        fs.clean.Filter("flow", method="savgol")
    with pytest.raises(ValidationError, match="needs a cutoff"):
        fs.clean.Filter("flow", method="butterworth")
    sf = synthetic.plant(seed=0, periods=50).signal()
    with pytest.raises(fs.SignalConfigError, match="longer than two samples"):
        fs.clean.Filter("flow", method="butterworth", cutoff="5min").transform(sf, {})


@pytest.mark.unit
def test_detotalize_unwraps_rollover_and_marks_resets():
    index = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
    total = pd.DataFrame({"m": [990.0, 995, 2, 7, np.nan, 17]}, index=index)
    sf = fs.SignalFrame(total, meta={"m": {"units": "m^3"}})
    out = fs.clean.Detotalize("m", rollover=1000, units="m^3/hr").transform(sf, {})
    assert out.values["m"].tolist()[1:4] == [5.0, 7.0, 5.0]
    assert out.values["m"].iloc[5] == 5.0  # 10 m^3 over the 2 h gap
    assert out.units["m"] == "m^3/hr"
    assert out.flags["m"].iloc[0] == Flag.MISSING
    reset = fs.clean.Detotalize("m").transform(sf, {})
    assert np.isnan(reset.values["m"].iloc[2])
    assert reset.flags["m"].iloc[2] == Flag.OUT_OF_RANGE
    per_minute = fs.clean.Detotalize("m", per="1min", rollover=1000).transform(sf, {})
    assert per_minute.values["m"].iloc[1] == pytest.approx(5 / 60)
