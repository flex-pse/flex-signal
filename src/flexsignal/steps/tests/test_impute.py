import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag
from flexsignal.testing import synthetic


def _run(step, sf):
    return step.transform(sf, step.fit(sf))


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interpolate_fills_exactly_the_short_gaps(seed):
    case = (
        synthetic.plant(seed=seed)
        .with_gaps("flow", count=5, length=2)  # 2 missing at 5 min: span 15 min
        .with_gaps("flow", count=3, length=20)  # span 105 min: too long
    )
    out = _run(fs.impute.Interpolate("flow", max_gap="15min"), case.signal())
    filled = ((out.flags["flow"] & int(Flag.IMPUTED_SHORT)) != 0).to_numpy()
    # Both gap injections share the "gap" kind; short gaps are the 2-sample runs.
    gaps = case.fault_mask("gap", "flow").to_numpy()
    runs = np.split(
        np.flatnonzero(gaps), np.flatnonzero(np.diff(np.flatnonzero(gaps)) > 1) + 1
    )
    short = np.zeros_like(gaps)
    for run in runs:
        if len(run) == 2:
            short[run] = True
    assert np.array_equal(filled, short)
    assert out.values["flow"][gaps & ~short].isna().all()
    error = (out.values["flow"][short] - case.clean["flow"][short]).abs()
    assert error.max() < 6 * case.noise["flow"]
    flags = out.flags["flow"].to_numpy()[short]
    assert np.all(flags == int(Flag.MISSING | Flag.IMPUTED_SHORT))


@pytest.mark.unit
@pytest.mark.parametrize("method", ["linear", "nearest", "previous", "pchip", "akima"])
def test_every_method_fills_a_linear_signal(method):
    index = pd.date_range("2025-01-01", periods=8, freq="1h", tz="UTC")
    x = np.arange(8, dtype=float)
    observed = x.copy()
    observed[3] = np.nan
    sf = fs.SignalFrame.from_frame(pd.DataFrame({"x": observed}, index=index))
    out = _run(fs.impute.Interpolate("x", max_gap="2h", method=method), sf)
    expected = {"linear": 3.0, "pchip": 3.0, "akima": 3.0, "previous": 2.0}
    value = out.values["x"].iloc[3]
    if method == "nearest":
        assert value in (2.0, 4.0)
    else:
        assert value == pytest.approx(expected[method])


@pytest.mark.unit
def test_edges_and_excluded_values_are_never_filled():
    index = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame(
            {"x": [np.nan, 1, np.nan, 3, 4, np.nan], "on": [1, 1, 0, 1, 1, 1]},
            index=index,
        )
    )
    sf = fs.select.Mask(keep_when="on == 1", streams="x").transform(sf, {})
    out = _run(fs.impute.Interpolate("x", max_gap="1D"), sf)
    assert out.values["x"].isna().tolist() == [True, False, True, False, False, True]
