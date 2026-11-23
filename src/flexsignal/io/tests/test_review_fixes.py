"""Regression tests for review findings in reading and writing files."""

import warnings

import numpy as np
import pandas as pd
import pytest

from flexsignal.core import Flag, NaiveTimestampWarning, SignalDataError, SignalFrame
from flexsignal.io import files, read_csv, read_parquet, write_csv, write_parquet


def _unsorted_with_flags(index):
    return pd.DataFrame(
        {"a": [3.0, 1.0, 2.0], "a__flags": np.array([8, 0, 0], "uint16")},
        index=index,
    )


@pytest.mark.unit
def test_read_csv_keeps_flags_on_their_rows_when_the_file_is_unsorted(tmp_path):
    path = tmp_path / "unsorted.csv"
    path.write_text(
        "timestamp,a,a__flags\n"
        "2025-01-01T02:00:00+00:00,3.0,8\n"
        "2025-01-01T00:00:00+00:00,1.0,0\n"
        "2025-01-01T01:00:00+00:00,2.0,0\n",
        encoding="utf-8",
    )
    sf = read_csv(path)
    assert sf.values["a"].tolist() == [1.0, 2.0, 3.0]
    assert sf.flags["a"].tolist() == [0, 0, int(Flag.OUTLIER)]


@pytest.mark.unit
def test_read_parquet_keeps_flags_on_their_rows_when_a_foreign_file_is_unsorted(
    tmp_path,
):
    index = pd.DatetimeIndex(
        ["2025-01-01T02:00Z", "2025-01-01T00:00Z", "2025-01-01T01:00Z"], name="t"
    )
    path = tmp_path / "unsorted.parquet"
    _unsorted_with_flags(index).to_parquet(path)
    sf = read_parquet(path)
    assert sf.values["a"].tolist() == [1.0, 2.0, 3.0]
    assert sf.flags["a"].tolist() == [0, 0, 8]


@pytest.mark.unit
def test_foreign_parquet_with_naive_index_flags_and_timezone(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "America/Los_Angeles")  # differs from timezone=
    index = pd.DatetimeIndex(["2025-01-01T00:00", "2025-01-01T01:00"], name="t")
    df = pd.DataFrame(
        {"a": [1.0, 2.0], "a__flags": np.array([0, 8], "uint16")}, index=index
    )
    path = tmp_path / "naive.parquet"
    df.to_parquet(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NaiveTimestampWarning)
        sf = read_parquet(path, timezone="UTC")
    assert sf.index[0] == pd.Timestamp("2025-01-01T00:00Z")
    assert sf.flags["a"].tolist() == [0, 8]


@pytest.mark.unit
def test_foreign_parquet_naive_index_and_flags_warns_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    index = pd.DatetimeIndex(["2025-01-01T00:00", "2025-01-01T01:00"], name="t")
    df = pd.DataFrame(
        {"a": [1.0, 2.0], "a__flags": np.array([0, 8], "uint16")}, index=index
    )
    path = tmp_path / "naive.parquet"
    df.to_parquet(path)
    with pytest.warns(NaiveTimestampWarning) as caught:
        sf = read_parquet(path)
    assert len([w for w in caught if w.category is NaiveTimestampWarning]) == 1
    assert sf.flags["a"].tolist() == [0, 8]


@pytest.mark.unit
def test_read_parquet_time_column_with_flags(tmp_path):
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01T01:00Z", "2025-01-01T00:00Z"]),
            "a": [2.0, 1.0],
            "a__flags": np.array([8, 0], "uint16"),
        }
    )
    path = tmp_path / "timecol.parquet"
    df.to_parquet(path, index=False)
    sf = read_parquet(path, time_column="ts")
    assert sf.columns == ["a"]
    assert sf.values["a"].tolist() == [1.0, 2.0]
    assert sf.flags["a"].tolist() == [0, 8]


@pytest.mark.unit
def test_read_parquet_names_a_missing_time_column(tmp_path):
    df = pd.DataFrame({"a": [1.0], "a__flags": np.array([0], "uint16")})
    path = tmp_path / "notime.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(SignalDataError, match="time column 'ts' not found"):
        read_parquet(path, time_column="ts")


@pytest.mark.unit
def test_read_csv_accepts_utc_offsets_that_change_across_dst(tmp_path):
    path = tmp_path / "dst.csv"
    path.write_text(
        "timestamp,a\n2025-03-09T01:00:00-08:00,1\n2025-03-09T03:00:00-07:00,2\n",
        encoding="utf-8",
    )
    sf = read_csv(path)
    assert list(sf.index) == [
        pd.Timestamp("2025-03-09T09:00Z"),
        pd.Timestamp("2025-03-09T10:00Z"),
    ]


@pytest.mark.unit
def test_read_csv_refuses_offsets_mixed_with_naive_timestamps(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(
        "timestamp,a\n2025-03-09T01:00:00-08:00,1\n2025-03-09T03:00:00,2\n",
        encoding="utf-8",
    )
    with pytest.raises(SignalDataError, match="UTC offset"):
        read_csv(path, timezone="UTC")


@pytest.mark.unit
def test_read_csv_unparseable_timestamps_are_a_data_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,a\nyesterday-ish,1\n", encoding="utf-8")
    with pytest.raises(SignalDataError, match="could not be parsed as timestamps"):
        read_csv(path, timezone="UTC")


@pytest.mark.unit
def test_empty_frame_round_trips_through_csv(tmp_path):
    sf = SignalFrame(
        pd.DataFrame(
            {"a": pd.Series([], dtype=float)}, index=pd.DatetimeIndex([], tz="UTC")
        )
    )
    path = write_csv(sf, tmp_path / "empty.csv", include_flags=True)
    assert read_csv(path).equals(sf)


@pytest.mark.unit
def test_write_csv_refuses_a_stream_named_like_the_time_column(tmp_path):
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    sf = SignalFrame(pd.DataFrame({"timestamp": [1.0, 2.0]}, index=index))
    with pytest.raises(SignalDataError, match="read back as 'timestamp.1'") as exc:
        write_csv(sf, tmp_path / "ts.csv")
    assert exc.value.field == "timestamp"
    assert not (tmp_path / "ts.csv").exists()


@pytest.mark.unit
def test_read_parquet_refuses_a_newer_format_version(tmp_path, monkeypatch):
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    sf = SignalFrame(pd.DataFrame({"a": [1.0, 2.0]}, index=index))
    monkeypatch.setattr(files, "FORMAT_VERSION", 2)
    path = write_parquet(sf, tmp_path / "future.parquet")
    monkeypatch.undo()
    with pytest.raises(SignalDataError, match="newer flex-signal"):
        read_parquet(path)
    # The current version still reads.
    assert read_parquet(write_parquet(sf, tmp_path / "now.parquet")).equals(sf)
