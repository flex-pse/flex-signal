"""The [foundation] and [t0] adapters: ChronosForecast and T0Forecast, zero-shot."""

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.backends import registered_backends
from flexsignal.core import SignalBackendError, SignalConfigError
from flexsignal.testing import synthetic

foundation = importlib.import_module("flexsignal.adapters.foundation")
SMALL = "autogluon/chronos-2-small"
T0 = "theforecastingcompany/t0-alpha"


def _hourly(days=10, seed=1):
    case = synthetic.plant(seed=seed, periods=days * 24, freq="1h")
    return case.signal().select(["flow", "temperature"])


def _price_load(days=10, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=24 * days, freq="1h", tz="UTC")
    price = 0.1 + 0.05 * rng.random(len(index))
    load = 50 - 200 * price + rng.normal(0, 0.1, len(index))
    truth = load[-24:].copy()
    load[-24:] = np.nan
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame({"load": load, "price": price}, index=index),
        units={"load": "kW", "price": "USD/kWh"},
    )
    return sf, truth


def _spec(kind="ChronosForecast", **options):
    step = {"kind": kind, "streams": ["flow"], "horizon": "1D"}
    return {
        "schema_version": "0.1.0",
        "name": "chronos",
        "steps": [{**step, **options}],
    }


@pytest.fixture
def weights():
    """Load Chronos-2 small, or skip when the extra or the weights aren't there."""
    pytest.importorskip("chronos")
    try:
        foundation._chronos(SMALL, foundation.PINNED_REVISIONS[SMALL])
    except SignalBackendError as exc:
        pytest.skip(f"Chronos-2 weights unavailable: {exc}")
    return SMALL


@pytest.fixture
def t0_weights():
    """Load t0-alpha, or skip when the [t0] extra or the weights aren't there."""
    pytest.importorskip("t0")
    try:
        foundation._t0(T0, foundation.PINNED_REVISIONS[T0])
    except SignalBackendError as exc:
        pytest.skip(f"t0-alpha weights unavailable: {exc}")
    return T0


# -- registration and options (no library, no network) ----------------------------


@pytest.mark.unit
def test_registered_with_documented_options():
    step = fs.forecast.ChronosForecast("flow", horizon="1D")
    assert step.model == "amazon/chronos-2"
    assert step.revision is None and step.context is None
    assert foundation.PINNED_REVISIONS["amazon/chronos-2"] == (
        "29ec3766d36d6f73f0696f85560a422f50e8498c"
    )
    fields = type(step).model_fields
    assert "8192" in fields["context"].description
    keys = {info.key for info in registered_backends()}
    assert {"chronos:chronos-2", "t0:t0-alpha"} <= keys
    assert "timesfm:timesfm-3.0" not in keys
    spec = fs.load_spec(_spec(model=SMALL, context="7D", quantiles=[0.1, 0.9]))
    assert fs.load_spec(fs.dump_spec(spec)) == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    ("option", "value"),
    [("model", "amazon/chronos-t5-large"), ("context", 7), ("revision", 3)],
)
def test_bad_options_fail_when_the_spec_loads(option, value):
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(_spec(**{option: value}))
    assert err.value.field == f"steps.0.{option}"


@pytest.mark.unit
def test_timesfm_needs_its_own_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "timesfm", None)
    with pytest.raises(SignalBackendError, match=r"pip install timesfm"):
        fs.run(_spec(model="google/timesfm-2.5-200m-pytorch"), _hourly(days=3))


@pytest.mark.unit
def test_no_model_with_a_restrictive_licence_is_offered():
    # Only open (Apache-2.0) weights ship; others are the user's to register.
    assert not hasattr(foundation, "LICENCES")
    for cls in (fs.forecast.ChronosForecast, fs.forecast.T0Forecast):
        assert "accept_licence" not in cls.model_fields
    with pytest.raises(SignalConfigError):
        fs.load_spec(_spec(model="google/timesfm-3.0-pytorch"))


@pytest.mark.unit
def test_a_missing_chronos_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "chronos", None)
    foundation._chronos.cache_clear()
    try:
        with pytest.raises(SignalBackendError, match=r"flex-signal\[foundation\]"):
            fs.run(_spec(), _hourly(days=3))
    finally:
        foundation._chronos.cache_clear()


# -- running the model --------------------------------------------------------------


@pytest.mark.integration
def test_runs_from_python_and_json_identically_and_saves(weights, tmp_path):
    sf = _hourly()
    step = fs.forecast.ChronosForecast(
        "flow", horizon="1D", quantiles=[0.1, 0.9], model=weights, context="7D"
    )
    from_python = fs.run(fs.Pipeline([step], name="chronos").spec, sf)
    spec = _spec(model=weights, context="7D", quantiles=[0.1, 0.9])
    from_json = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    again = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(again.data, from_json.data)

    out = from_json.signal
    ahead = out.has_flag(fs.Flag.FORECAST)["flow"]
    assert ahead.sum() == 24
    lo, mid, hi = (out.values[c][ahead] for c in ("flow__q10", "flow", "flow__q90"))
    assert (lo <= mid).all() and (mid <= hi).all() and (hi > lo).any()
    assert 40 < mid.mean() < 160  # the plant's flow is around 100

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved)
    pd.testing.assert_frame_equal(loaded.transform(sf).values, out.values)


@pytest.mark.integration
def test_uses_exog_as_covariates(weights):
    sf, truth = _price_load()
    with_price = fs.forecast.ChronosForecast(
        "load", horizon="1D", exog=["price"], model=weights
    )
    alone = fs.forecast.ChronosForecast("load", horizon="1D", model=weights)
    errors = {}
    for name, step in {"with": with_price, "alone": alone}.items():
        result = fs.run(fs.Pipeline([step]).spec, sf).signal
        errors[name] = np.abs(result.values["load"].to_numpy()[-24:] - truth).mean()
    # The load is driven by the price, which is random: only the covariate helps.
    assert errors["with"] < 0.5 * errors["alone"], errors


@pytest.mark.integration
def test_beats_persistence_on_daily_cycles(weights):
    sf = _hourly(days=12)
    step = fs.forecast.ChronosForecast(
        "flow", horizon="1D", quantiles=[0.1, 0.9], model=weights
    )
    result = fs.backtest(
        fs.Pipeline([step]).spec,
        sf,
        folds=3,
        compare=[
            fs.forecast.Persistence("flow", horizon="1D"),
            fs.forecast.SeasonalNaive("flow", horizon="1D"),
        ],
    )
    summary = result.summary().set_index("method")
    print(summary)
    assert summary.loc["ChronosForecast", "skill"] > 0.3
    assert summary.loc["ChronosForecast", "coverage"] > 0.5


# -- t0-alpha ([t0]) ------------------------------------------------------------------


@pytest.mark.unit
def test_t0_registered_with_a_pinned_revision():
    step = fs.forecast.T0Forecast("flow", horizon="1D")
    assert step.model == T0 and step.revision is None
    assert foundation.PINNED_REVISIONS[T0] == "741927847fe02367624749bb9290b2f09f128e7d"
    spec = fs.load_spec(_spec("T0Forecast", context="7D", quantiles=[0.1, 0.9]))
    assert fs.load_spec(fs.dump_spec(spec)) == spec
    with pytest.raises(SignalConfigError):
        fs.load_spec(_spec("T0Forecast", model="theforecastingcompany/t0-gamma"))


@pytest.mark.unit
def test_a_missing_t0_names_its_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "t0", None)
    foundation._t0.cache_clear()
    try:
        with pytest.raises(SignalBackendError, match=r"flex-signal\[t0\]"):
            fs.run(_spec("T0Forecast"), _hourly(days=3))
    finally:
        foundation._t0.cache_clear()


@pytest.mark.integration
def test_t0_forecasts_saves_and_repeats(t0_weights, tmp_path):
    sf = _hourly()
    spec = _spec("T0Forecast", context="7D", quantiles=[0.1, 0.9])
    first = fs.run(json.dumps(spec), sf)
    again = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(first.data, again.data)
    out = first.signal
    ahead = out.has_flag(fs.Flag.FORECAST)["flow"]
    assert ahead.sum() == 24
    lo, mid, hi = (out.values[c][ahead] for c in ("flow__q10", "flow", "flow__q90"))
    assert (lo <= mid).all() and (mid <= hi).all() and (hi > lo).any()
    assert 40 < mid.mean() < 160
    loaded = fs.FittedPipeline.load(first.fitted.save(tmp_path / "fitted"))
    pd.testing.assert_frame_equal(loaded.transform(sf).values, out.values)


@pytest.mark.integration
def test_t0_uses_exog_as_covariates(t0_weights):
    sf, truth = _price_load()
    errors = {}
    for name, exog in {"with": ["price"], "alone": []}.items():
        step = fs.forecast.T0Forecast("load", horizon="1D", exog=exog)
        result = fs.run(fs.Pipeline([step]).spec, sf).signal
        errors[name] = np.abs(result.values["load"].to_numpy()[-24:] - truth).mean()
    assert errors["with"] < 0.5 * errors["alone"], errors


@pytest.mark.integration
def test_t0_beats_persistence_on_daily_cycles(t0_weights):
    step = fs.forecast.T0Forecast("flow", horizon="1D", quantiles=[0.1, 0.9])
    result = fs.backtest(
        fs.Pipeline([step]).spec,
        _hourly(days=12),
        folds=3,
        compare=[fs.forecast.Persistence("flow", horizon="1D")],
    )
    summary = result.summary().set_index("method")
    print(summary)
    assert summary.loc["T0Forecast", "skill"] > 0.3
    assert summary.loc["T0Forecast", "coverage"] > 0.5
