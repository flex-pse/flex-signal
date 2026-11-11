import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import NaiveTimestampWarning, SignalDataError
from flexsignal.io import read_csv, read_parquet, write_csv, write_parquet
from flexsignal.testing import synthetic


def _cleaned():
    case = (
        synthetic.plant(seed=5, periods=300)
        .with_spikes("flow", count=3)
        .with_gaps("flow", count=2)
    )
    pipe = fs.Pipeline(
        [fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0)], name="io"
    )
    return fs.run(pipe.spec, case.signal()).signal


@pytest.mark.unit
def test_parquet_round_trip_keeps_values_flags_meta_and_history(tmp_path):
    sf = _cleaned()
    back = read_parquet(write_parquet(sf, tmp_path / "x.parquet"))
    assert back.equals(sf)
    assert back.history == sf.history and back.units["flow"] == "m^3/hr"


@pytest.mark.unit
def test_plain_parquet_from_other_tools(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"a": [1.0, None]},
        index=pd.date_range("2025-01-01", periods=2, freq="1h", name="t"),
    )
    df.to_parquet(tmp_path / "plain.parquet")
    monkeypatch.setenv("TZ", "America/Denver")
    with pytest.warns(NaiveTimestampWarning, match="America/Denver"):
        local = read_parquet(tmp_path / "plain.parquet")
    assert local.index[0] == pd.Timestamp("2025-01-01 07:00", tz="UTC")
    sf = read_parquet(tmp_path / "plain.parquet", timezone="UTC", units={"a": "bar"})
    assert sf.units == {"a": "bar"} and sf.flag_counts()["a"]["flags"] == {"MISSING": 1}


@pytest.mark.unit
def test_csv_round_trip_is_exact(tmp_path):
    sf = _cleaned()
    path = write_csv(sf, tmp_path / "x.csv", include_flags=True)
    back = read_csv(path, units=sf.units)
    assert back.equals(sf)


@pytest.mark.unit
def test_csv_naive_timestamps_and_bom(tmp_path, monkeypatch):
    path = tmp_path / "naive.csv"
    path.write_bytes(
        "\ufefftimestamp,x\n2025-01-01 00:00:00,1.5\n2025-01-01 01:00:00,2.5\n".encode()
    )
    monkeypatch.setenv("TZ", "UTC")
    with pytest.warns(NaiveTimestampWarning, match="'UTC'"):
        read_csv(path)
    sf = read_csv(path, timezone="UTC")
    assert sf.values["x"].tolist() == [1.5, 2.5]
    with pytest.raises(SignalDataError, match="no 'time'"):
        read_csv(path, time_column="time")
