"""Regression tests for review findings in the core layer."""

import numpy as np
import pandas as pd
import pytest

from flexsignal.core import (
    Flag,
    SignalConfigError,
    SignalDataError,
    SignalFrame,
    evaluate_condition,
    evaluate_formula,
)
from flexsignal.core.times import check_duration, check_timestamp

DST_STAMPS = ["2025-03-09T01:00:00-08:00", "2025-03-09T03:00:00-07:00"]
DST_UTC = [pd.Timestamp("2025-03-09T09:00Z"), pd.Timestamp("2025-03-09T10:00Z")]


def _flow():
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    return SignalFrame(pd.DataFrame({"flow": [1.0, 2.0, np.nan, 4.0]}, index=index))


# -- timestamps with changing UTC offsets --------------------------------------------


@pytest.mark.unit
def test_from_frame_time_column_with_offsets_that_change_across_dst():
    df = pd.DataFrame({"t": DST_STAMPS, "a": [1.0, 2.0]})
    sf = SignalFrame.from_frame(df, time_column="t")
    assert list(sf.index) == DST_UTC
    assert sf.columns == ["a"]


@pytest.mark.unit
def test_from_frame_string_index_with_offsets_that_change_across_dst():
    sf = SignalFrame.from_frame(pd.DataFrame({"a": [1.0, 2.0]}, index=DST_STAMPS))
    assert list(sf.index) == DST_UTC


@pytest.mark.unit
def test_offsets_mixed_with_naive_timestamps_are_refused():
    df = pd.DataFrame({"t": [DST_STAMPS[0], "2025-03-09T03:00:00"], "a": [1.0, 2.0]})
    with pytest.raises(SignalDataError, match="carry a UTC offset"):
        SignalFrame.from_frame(df, time_column="t", timezone="UTC")


@pytest.mark.unit
def test_an_unparseable_string_index_says_what_failed():
    df = pd.DataFrame({"a": [1.0]}, index=["not a time"])
    with pytest.raises(SignalDataError, match="the index could not be parsed") as exc:
        SignalFrame.from_frame(df)
    assert "ISO 8601" in exc.value.message


# -- integer constants in expressions ------------------------------------------------


@pytest.mark.unit
def test_integer_constants_behave_like_floats():
    sf = _flow()
    np.testing.assert_allclose(
        evaluate_formula(sf, "flow * 10 ** -3"), [0.001, 0.002, np.nan, 0.004]
    )
    assert evaluate_condition(sf, "flow < 2 ** 70").tolist() == [
        True,
        True,
        False,
        True,
    ]
    result = evaluate_formula(sf, "flow + 7 // 0")
    assert np.isinf(result[[0, 1, 3]]).all()


# -- durations and timestamps --------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text", ["NaT", "nat", "nan", ""])
def test_a_missing_duration_is_refused(text):
    with pytest.raises(ValueError, match="is not a duration"):
        check_duration(text)
    with pytest.raises(ValueError, match="is not a duration"):
        check_duration(text, allow_zero=True)
    with pytest.raises(ValueError, match="is not a duration"):
        check_duration(text, signed=True)


@pytest.mark.unit
def test_check_duration_zero_and_sign_rules():
    assert check_duration("0s", allow_zero=True) == pd.Timedelta(0)
    with pytest.raises(ValueError, match="must be positive"):
        check_duration("0s")
    with pytest.raises(ValueError, match="0 or more"):
        check_duration("-1h", allow_zero=True)
    assert check_duration("-5min", signed=True) == pd.Timedelta("-5min")


@pytest.mark.unit
@pytest.mark.parametrize("text", ["now", "today", "Yesterday", " tomorrow "])
def test_relative_timestamps_are_refused(text):
    with pytest.raises(ValueError, match="relative to the clock"):
        check_timestamp(text)


@pytest.mark.unit
def test_a_missing_timestamp_is_refused():
    with pytest.raises(ValueError, match="is not a timestamp"):
        check_timestamp("NaT")
    assert check_timestamp("2025-01-01T00:00:00Z") == pd.Timestamp("2025-01-01T00:00Z")


@pytest.mark.unit
def test_held_refuses_a_nat_duration():
    with pytest.raises(SignalConfigError, match="'NaT' is not a duration"):
        evaluate_condition(_flow(), 'held(flow > 0, "NaT")')


@pytest.mark.unit
def test_held_on_a_frame_without_timestamps_is_a_config_error():
    with pytest.raises(SignalConfigError, match="DatetimeIndex"):
        evaluate_condition(pd.DataFrame({"c": [0.0, 0.0]}), 'held(c == 0, "1h")')


# -- backticked names ----------------------------------------------------------------


@pytest.mark.unit
def test_a_backticked_name_may_contain_at_sign():
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    sf = SignalFrame(pd.DataFrame({"flow@site": [1.0, 2.0]}, index=index))
    assert evaluate_condition(sf, "`flow@site` > 1").tolist() == [False, True]


@pytest.mark.unit
def test_an_unquoted_at_sign_is_still_refused():
    with pytest.raises(SignalConfigError, match="'@' variable references"):
        evaluate_condition(_flow(), "flow > @limit")


# -- frames --------------------------------------------------------------------------


@pytest.mark.unit
def test_an_all_none_stream_becomes_missing_floats():
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    sf = SignalFrame(pd.DataFrame({"a": [None, None]}, index=index))
    assert sf.values["a"].dtype == "float64"
    assert sf.values["a"].isna().all()
    assert (sf.flags["a"] == int(Flag.MISSING)).all()


@pytest.mark.unit
def test_an_empty_object_stream_is_accepted():
    sf = SignalFrame(
        pd.DataFrame({"a": pd.Series([], dtype=object)}, index=pd.DatetimeIndex([]))
    )
    assert len(sf) == 0
    assert sf.values["a"].dtype == "float64"


@pytest.mark.unit
def test_describe_flags_at_a_missing_timestamp_is_a_data_error():
    with pytest.raises(SignalDataError, match="no sample at 2030") as exc:
        _flow().describe_flags("flow", "2030-01-01")
    assert exc.value.field == "flow"
    with pytest.raises(SignalDataError, match="no stream named"):
        _flow().describe_flags("nope", "2025-01-01")
    assert _flow().describe_flags("flow", "2025-01-01T02:00Z") == ["MISSING"]
