"""SignalFrame edge cases: what is refused, and the smaller accessors."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalDataError, StreamMeta, local_timezone
from flexsignal.core.frame import FLAGS_SUFFIX


def _df(n=4):
    index = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0][:n], "b": [1.0, 2.0, 3.0, 4.0][:n]},
        index=index,
    )


# -- time zones ----------------------------------------------------------------------


@pytest.mark.unit
def test_an_unreadable_system_zone_falls_back_to_utc(monkeypatch):
    def broken():
        raise RuntimeError("no zone on this system")

    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr("flexsignal.core.frame.tzlocal.get_localzone_name", broken)
    assert local_timezone() == "UTC"


# -- construction: what is refused ---------------------------------------------------


@pytest.mark.unit
def test_values_must_be_a_dataframe():
    with pytest.raises(SignalDataError, match="values must be a pandas DataFrame"):
        fs.SignalFrame(_df()["a"])
    with pytest.raises(SignalDataError, match="expected a pandas DataFrame, got list"):
        fs.SignalFrame.from_frame([1.0, 2.0])


@pytest.mark.unit
def test_the_index_must_be_a_datetime_index():
    with pytest.raises(SignalDataError, match="must be a pandas DatetimeIndex"):
        fs.SignalFrame(_df().reset_index(drop=True))


@pytest.mark.unit
def test_stream_names_must_be_unique_strings():
    numbered = _df()
    numbered.columns = [0, 1]
    with pytest.raises(SignalDataError, match="stream names must be strings, got 0"):
        fs.SignalFrame(numbered)
    doubled = _df()
    doubled.columns = ["a", "a"]
    with pytest.raises(SignalDataError, match=r"duplicate stream names \['a'\]"):
        fs.SignalFrame(doubled)


@pytest.mark.unit
def test_flags_must_match_the_values():
    df = _df()
    flags = pd.DataFrame(0, index=df.index, columns=df.columns, dtype="uint16")
    assert fs.SignalFrame(df, flags).flags.eq(0).all().all()
    with pytest.raises(SignalDataError, match="flags must be a pandas DataFrame"):
        fs.SignalFrame(df, flags.to_numpy())
    with pytest.raises(SignalDataError, match="same columns"):
        fs.SignalFrame(df, flags[["b", "a"]])
    with pytest.raises(SignalDataError, match="same index"):
        fs.SignalFrame(df, flags.iloc[:3])
    shifted = flags.set_axis(flags.index + pd.Timedelta("1min"), axis=0)
    with pytest.raises(SignalDataError, match="same index"):
        fs.SignalFrame(df, shifted)


# -- from_frame ----------------------------------------------------------------------


@pytest.mark.unit
def test_from_frame_names_a_missing_time_column():
    with pytest.raises(SignalDataError, match="time column 'ts' not found"):
        fs.SignalFrame.from_frame(_df(), time_column="ts")


@pytest.mark.unit
def test_from_frame_parses_a_text_index():
    text = _df().set_axis(
        [
            "2025-01-01 00:00",
            "2025-01-01 00:15",
            "2025-01-01 00:30",
            "2025-01-01 00:45",
        ],
        axis=0,
    )
    sf = fs.SignalFrame.from_frame(text, timezone="UTC")
    assert sf.index[1] == pd.Timestamp("2025-01-01 00:15", tz="UTC")
    # The first entry fixes the format, so the junk raises without a warning.
    junk = _df().set_axis(["2025-01-01 00:00", "soon", "later", "never"], axis=0)
    with pytest.raises(SignalDataError, match="could not be parsed as timestamps"):
        fs.SignalFrame.from_frame(junk, timezone="UTC")


@pytest.mark.unit
def test_from_frame_merges_meta_objects_and_units():
    sf = fs.SignalFrame.from_frame(
        _df(),
        meta={"a": StreamMeta(units="bar"), "b": {"units": "m"}},
        units={"b": "m^3/hr"},
    )
    assert sf.units == {"a": "bar", "b": "m^3/hr"}


# -- access --------------------------------------------------------------------------


@pytest.mark.unit
def test_repr_shows_the_span_or_empty():
    sf = fs.SignalFrame.from_frame(_df())
    assert repr(sf) == (
        "SignalFrame(4 rows × 2 streams, "
        "2025-01-01T00:00:00+00:00 → 2025-01-01T00:45:00+00:00)"
    )
    assert repr(fs.SignalFrame.from_frame(_df(n=0))) == (
        "SignalFrame(0 rows × 2 streams, empty)"
    )


@pytest.mark.unit
def test_stream_and_select_name_missing_streams():
    sf = fs.SignalFrame.from_frame(_df())
    assert sf.stream("b").tolist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(SignalDataError, match="no stream named 'zzz'") as err:
        sf.stream("zzz")
    assert err.value.field == "zzz"
    with pytest.raises(SignalDataError, match=r"no streams named \['zzz'\]"):
        sf.select(["a", "zzz"])


@pytest.mark.unit
def test_valid_has_flag_and_history():
    sf = fs.SignalFrame.from_frame(_df()).with_record({"step": "x"})
    assert sf.valid()["a"].tolist() == [True, False, True, True]
    assert sf.has_flag(Flag.MISSING)["a"].tolist() == [False, True, False, False]
    assert sf.history == ({"step": "x"},)


@pytest.mark.unit
def test_to_pandas_with_and_without_flags():
    sf = fs.SignalFrame.from_frame(_df())
    assert list(sf.to_pandas().columns) == ["a", "b"]
    flat = sf.to_pandas(include_flags=True)
    assert list(flat.columns) == ["a", "b", f"a{FLAGS_SUFFIX}", f"b{FLAGS_SUFFIX}"]
    assert flat[f"a{FLAGS_SUFFIX}"].tolist() == [0, int(Flag.MISSING), 0, 0]


@pytest.mark.unit
def test_an_index_of_plain_text_gets_flex_signals_error_not_a_pandas_warning():
    import pandas as pd

    import flexsignal as fs

    with pytest.raises(fs.SignalDataError, match="could not be parsed as timestamps"):
        fs.SignalFrame.from_frame(pd.DataFrame({"flow": [1.0, 2.0]}, index=["a", "b"]))
