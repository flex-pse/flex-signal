import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from flexsignal.core import (
    Flag,
    NaiveTimestampWarning,
    SignalConfigError,
    SignalDataError,
    SignalError,
    SignalFrame,
    StreamMeta,
    evaluate_condition,
    flag_names,
    local_timezone,
    parse_flags,
)


def _df(tz="UTC", n=4):
    index = pd.date_range("2025-01-01", periods=n, freq="15min", tz=tz)
    return pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0][:n], "b": [1, 2, 3, 4][:n]}, index=index
    )


@pytest.mark.unit
def test_from_frame_flags_missing_and_converts_to_float_utc():
    sf = SignalFrame.from_frame(_df(tz="America/Los_Angeles"), units={"a": "m^3/hr"})
    assert str(sf.index.tz) == "UTC"
    assert sf.index.dtype == "datetime64[ns, UTC]"
    assert sf.values["b"].dtype == "float64"
    assert flag_names(int(sf.flags["a"].iloc[1])) == ["MISSING"]
    assert sf.flags["b"].eq(0).all()
    assert sf.units == {"a": "m^3/hr", "b": None}


@pytest.mark.unit
def test_naive_index_is_taken_as_local_time_with_a_warning(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    df = _df(tz=None)
    with pytest.warns(NaiveTimestampWarning, match="'America/Los_Angeles'") as caught:
        sf = SignalFrame.from_frame(df)
    assert len(caught) == 1
    assert caught[0].filename == __file__  # points at the caller, not flex-signal
    assert sf.index[0] == pd.Timestamp("2025-01-01 08:00", tz="UTC")
    # The constructor follows the same rule.
    with pytest.warns(NaiveTimestampWarning):
        assert SignalFrame(df).equals(sf)


@pytest.mark.unit
def test_an_explicit_timezone_wins_and_does_not_warn(monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf = SignalFrame.from_frame(_df(tz=None), timezone="UTC")
        aware = SignalFrame.from_frame(_df(tz="Europe/Berlin"))
    assert sf.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert aware.index[0] == pd.Timestamp("2024-12-31 23:00", tz="UTC")


@pytest.mark.unit
def test_daylight_saving_gaps_still_fail(monkeypatch):
    spring_forward = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.date_range("2025-03-09 01:00", periods=2, freq="1h"),
    )
    with pytest.raises(SignalDataError, match="daylight-saving"):
        SignalFrame.from_frame(spring_forward, timezone="America/Los_Angeles")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    with pytest.warns(NaiveTimestampWarning):
        with pytest.raises(SignalDataError, match="this machine's time zone"):
            SignalFrame.from_frame(spring_forward)


@pytest.mark.unit
def test_local_timezone(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    assert local_timezone() == "Asia/Tokyo"
    monkeypatch.setenv("TZ", "Not/AZone")  # unknown: fall through to the system
    monkeypatch.setattr("flexsignal.core.frame._system_timezone", lambda: None)
    assert local_timezone() == "UTC"
    monkeypatch.delenv("TZ")
    monkeypatch.setattr(
        "flexsignal.core.frame._system_timezone", lambda: "America/Chicago"
    )
    assert local_timezone() == "America/Chicago"


@pytest.mark.unit
def test_duplicate_timestamps_are_rejected():
    df = _df()
    df.index = df.index[[0, 0, 1, 2]]
    with pytest.raises(SignalDataError, match="duplicate timestamps"):
        SignalFrame.from_frame(df)


@pytest.mark.unit
def test_unsorted_index_is_sorted_with_flags():
    df = _df().iloc[::-1]
    sf = SignalFrame.from_frame(df)
    assert sf.index.is_monotonic_increasing
    assert np.isnan(sf.values["a"].iloc[1])
    assert sf.flags["a"].iloc[1] == Flag.MISSING


@pytest.mark.unit
def test_non_numeric_stream_is_rejected():
    df = _df().assign(c=["x", "y", "z", "w"])
    with pytest.raises(SignalDataError, match="not numeric"):
        SignalFrame.from_frame(df)


@pytest.mark.unit
def test_time_column():
    df = _df().reset_index(names="ts")
    sf = SignalFrame.from_frame(df, time_column="ts")
    assert sf.columns == ["a", "b"]


@pytest.mark.unit
def test_values_are_copies():
    sf = SignalFrame.from_frame(_df())
    values = sf.values
    values.iloc[0, 0] = 99.0
    assert sf.values.iloc[0, 0] == 1.0


@pytest.mark.unit
def test_meta_for_unknown_stream_is_rejected():
    with pytest.raises(SignalDataError, match="don't exist"):
        SignalFrame(_df(), meta={"zzz": StreamMeta(units="bar")})


@pytest.mark.unit
def test_flag_counts_select_equals_and_pickle():
    sf = SignalFrame.from_frame(_df(), units={"a": "bar"})
    counts = sf.flag_counts()
    assert counts["a"] == {"rows": 4, "valid": 3, "flags": {"MISSING": 1}}
    assert sf.select(["b"]).columns == ["b"]
    clone = pickle.loads(pickle.dumps(sf))
    assert clone.equals(sf)
    assert sf.describe_flags("a", "2025-01-01 00:15") == ["MISSING"]


@pytest.mark.unit
def test_flag_helpers():
    assert parse_flags(["OUTLIER", "STUCK"]) == Flag.OUTLIER | Flag.STUCK
    with pytest.raises(ValueError, match="unknown flag"):
        parse_flags("NOPE")
    assert flag_names(int(Flag.MISSING | Flag.RESAMPLED)) == ["MISSING", "RESAMPLED"]


@pytest.mark.unit
def test_errors_carry_field_and_pickle():
    err = SignalConfigError("bad", field="steps.1.window", value="x")
    assert str(err) == "steps.1.window: bad"
    clone = pickle.loads(pickle.dumps(err))
    assert isinstance(clone, SignalError) and clone.field == "steps.1.window"


@pytest.mark.unit
def test_condition_evaluation():
    values = _df().astype(float)
    mask = evaluate_condition(values, "a > 1 and b < 4")
    assert mask.tolist() == [False, False, True, False]
    with pytest.raises(SignalConfigError, match="unknown stream 'zzz'") as info:
        evaluate_condition(values, "zzz > 1", field="keep_when")
    assert info.value.field == "keep_when"
    with pytest.raises(SignalConfigError, match="'@'"):
        evaluate_condition(values, "a > @limit")
