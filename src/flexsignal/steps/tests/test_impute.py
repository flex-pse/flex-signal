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


# -- model imputers and tiered --------------------------------------------------------


def _gappy(seed=0):
    return (
        synthetic.plant(seed=seed)
        .with_gaps("flow", count=4, length=2)
        .with_gaps("flow", count=3, length=24)
    )


def _gap_error(out, case):
    gaps = case.fault_mask("gap", "flow").to_numpy()
    filled = out.values["flow"].to_numpy()[gaps]
    assert not np.isnan(filled).any()
    error = filled - case.clean["flow"].to_numpy()[gaps]
    return float(np.sqrt(np.mean(error**2)))


@pytest.mark.component
@pytest.mark.parametrize(
    ("step", "limit"),
    [
        (fs.impute.Kalman("flow", seasonal=[{"period": "1D", "harmonics": 2}]), 2.0),
        (fs.impute.Regression("flow", features=["pressure"]), 3.0),
        (fs.impute.Regression("flow", fourier=[{"period": "1D"}]), 4.0),
        (fs.impute.KNN("flow", features=["pressure", "temperature"]), 3.5),
        (fs.impute.Iterative("flow", features=["pressure", "temperature"]), 3.0),
    ],
    ids=["kalman", "regression", "fourier", "knn", "iterative"],
)
def test_model_imputers_fill_every_gap_accurately(step, limit):
    case = _gappy()
    fitted, out = fs.Pipeline([step]).fit_transform(case.signal())
    assert _gap_error(out, case) < limit
    gaps = case.fault_mask("gap", "flow").to_numpy()
    assert np.all(
        out.flags["flow"].to_numpy()[gaps] == int(Flag.MISSING | Flag.IMPUTED_MODEL)
    )
    # Values that were present are untouched.
    kept = ~gaps
    assert np.array_equal(
        out.values["flow"].to_numpy()[kept], case.observed["flow"].to_numpy()[kept]
    )
    # The fitted state is JSON and reproduces the result.
    again = fs.FittedPipeline.from_payload(fitted.to_payload()).transform(case.signal())
    assert again.equals(out)


@pytest.mark.unit
def test_model_imputers_respect_max_gap_and_exclusions():
    index = pd.date_range("2025-01-01", periods=12, freq="1h", tz="UTC")
    x = np.arange(12, dtype=float)
    x[[0, 3, 6, 7, 8, 11]] = np.nan
    on = np.ones(12)
    on[3] = 0
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame({"x": x, "y": np.arange(12.0), "on": on}, index=index)
    )
    sf = fs.select.Mask(keep_when="on == 1", streams="x").transform(sf, {})
    step = fs.impute.Regression(
        "x", features=["y"], max_gap="2h", model="sklearn:LinearRegression"
    )
    out = _run(step, sf)
    # 3 is excluded; 6-8 is a 4h gap; 0 and 11 are edges (no length).
    assert out.values["x"].isna().tolist() == [
        True, False, False, True, False, False, True, True, True, False, False, True
    ]  # fmt: skip
    every_gap = _run(step.model_copy(update={"max_gap": None}), sf)
    assert every_gap.values["x"].isna().tolist() == [False] * 3 + [True] + [False] * 8
    assert every_gap.values["x"].iloc[11] == pytest.approx(11.0)


@pytest.mark.unit
def test_tiered_routes_gaps_by_length():
    case = _gappy(seed=1)
    step = fs.impute.Tiered(
        "flow",
        tiers=[
            {"max_gap": "15min", "method": "interpolate"},
            {
                "max_gap": None,
                "method": "regression",
                "options": {"features": ["pressure"]},
            },
        ],
    )
    fitted, out = fs.Pipeline([step]).fit_transform(case.signal())
    flags = out.flags["flow"].to_numpy()
    short = (flags & int(Flag.IMPUTED_SHORT)) != 0
    model = (flags & int(Flag.IMPUTED_MODEL)) != 0
    assert short.sum() == 4 * 2 and model.sum() == 3 * 24
    assert not (short & model).any()
    assert list(fitted.states[0]) == ["tiers"]
    assert fitted.states[0]["tiers"][0] == {}


@pytest.mark.unit
def test_tiered_and_regression_reject_bad_options():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="increasing max_gap"):
        fs.impute.Tiered(
            "x",
            tiers=[
                {"max_gap": "1h", "method": "interpolate"},
                {"max_gap": "15min", "method": "interpolate"},
            ],
        )
    with pytest.raises(ValidationError, match="only the last tier"):
        fs.impute.Tiered(
            "x",
            tiers=[
                {"max_gap": None, "method": "kalman"},
                {"max_gap": "1h", "method": "interpolate"},
            ],
        )
    with pytest.raises(ValidationError, match="tier 0 .interpolate."):
        fs.impute.Tiered("x", tiers=[{"max_gap": None, "method": "interpolate"}])
    with pytest.raises(ValidationError, match="tier 0 .regression."):
        fs.impute.Tiered("x", tiers=[{"max_gap": "1h", "method": "regression"}])
    with pytest.raises(ValidationError, match="both filled and used"):
        fs.impute.Regression("x", features=["x"])
    with pytest.raises(ValidationError, match="linear model"):
        fs.impute.Regression("x", features=["y"], model="scipy:curve_fit")
