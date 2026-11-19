import pandas as pd
import pytest

from flexsignal.core import Flag, SignalDataError
from flexsignal.io import read_csv, read_parquet, write_parquet
from flexsignal.testing import synthetic


def _plain():
    return pd.DataFrame(
        {"a": [1.0, None, 3.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC", name="t"),
    )


@pytest.mark.unit
def test_missing_files_are_data_errors(tmp_path):
    with pytest.raises(SignalDataError, match="file not found") as exc:
        read_parquet(tmp_path / "nope.parquet")
    assert "nope.parquet" in exc.value.message
    with pytest.raises(SignalDataError, match="file not found") as exc:
        read_csv(tmp_path / "nope.csv")
    assert "nope.csv" in exc.value.message


@pytest.mark.unit
def test_flags_column_without_its_stream_is_refused(tmp_path):
    path = tmp_path / "orphan.csv"
    path.write_text(
        "timestamp,a,b__flags\n2025-01-01T00:00:00+00:00,1.0,0\n", encoding="utf-8"
    )
    with pytest.raises(SignalDataError, match=r"without a matching stream: \['b'\]"):
        read_csv(path)


@pytest.mark.unit
def test_plain_parquet_with_flags_columns_keeps_the_flags(tmp_path):
    df = _plain()
    df["a__flags"] = pd.array([0, int(Flag.MANUAL), 0], dtype="uint16")
    df.to_parquet(tmp_path / "flags.parquet")
    sf = read_parquet(tmp_path / "flags.parquet")
    assert list(sf.values.columns) == ["a"]
    assert sf.flags["a"].iloc[1] & Flag.MANUAL
    assert sf.flag_counts()["a"]["flags"]["MANUAL"] == 1


@pytest.mark.unit
def test_units_override_those_stored_in_a_flexsignal_parquet(tmp_path):
    sf = synthetic.plant(seed=2, periods=20).signal()
    path = write_parquet(sf, tmp_path / "x.parquet")
    back = read_parquet(path, units={"flow": "L/s"})
    assert back.units["flow"] == "L/s"
    others = {k: v for k, v in sf.units.items() if k != "flow"}
    assert {k: back.units[k] for k in others} == others
