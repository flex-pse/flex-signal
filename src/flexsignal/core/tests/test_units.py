"""Unit conversion with pint (the [units] extra)."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError
from flexsignal.core import units as units_module
from flexsignal.core.exceptions import SignalBackendError
from flexsignal.core.units import convert


@pytest.mark.unit
def test_convert_scales_and_handles_offsets_and_nan():
    pytest.importorskip("pint")
    assert convert(3600.0, "m^3/hr", "m^3/s") == pytest.approx(1.0)
    assert convert(20.0, "degC", "K") == pytest.approx(293.15)
    out = convert(np.array([1.0, np.nan]), "kWh", "MJ")
    assert out[0] == pytest.approx(3.6)
    assert np.isnan(out[1])
    with pytest.raises(SignalConfigError, match="can't convert"):
        convert(1.0, "m^3/hr", "kW")
    with pytest.raises(SignalConfigError, match="can't read the units"):
        convert(1.0, "furlongs per blorp", "m")


@pytest.mark.unit
def test_missing_pint_names_the_extra(monkeypatch):
    units_module._registry.cache_clear()
    monkeypatch.setitem(__import__("sys").modules, "pint", None)
    try:
        with pytest.raises(SignalBackendError, match=r"flex-signal\[units\]"):
            convert(1.0, "m", "km")
    finally:
        units_module._registry.cache_clear()


@pytest.mark.unit
def test_convert_units_step_rescales_and_relabels():
    pytest.importorskip("pint")
    index = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"flow": [3.6, np.nan, 7.2], "level": [1.0, 2.0, 3.0]}, index=index
    )
    sf = fs.SignalFrame.from_frame(df, units={"flow": "m^3/hr", "level": "m"})
    step = fs.select.ConvertUnits("flow", to="L/s")
    out = fs.run(fs.Pipeline([step]).spec, sf).signal
    assert out.units["flow"] == "L/s"
    assert out.values["flow"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(out.values["flow"].iloc[1])
    pd.testing.assert_frame_equal(out.flags, sf.flags)
    assert out.values["level"].equals(sf.values["level"])
    spec = {
        "schema_version": "0.1.0",
        "name": "x",
        "steps": [{"kind": "convert_units", "streams": ["flow"], "to": "L/s"}],
    }
    pd.testing.assert_frame_equal(fs.run(spec, sf).data, out.values)
    no_units = fs.SignalFrame.from_frame(df[["level"]])
    with pytest.raises(SignalConfigError, match="no units to convert from"):
        fs.select.ConvertUnits("level", to="m").transform(no_units, {})
