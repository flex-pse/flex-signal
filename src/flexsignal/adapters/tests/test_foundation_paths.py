"""The foundation adapters' code paths, run against small stand-in libraries.

TimesFM isn't installed in any CI job, and t0 only in its own, so these tests
put minimal stand-ins for ``timesfm`` and ``t0`` in ``sys.modules``. They follow
the libraries' documented call shapes, so the adapter's handling of inputs and
outputs is checked without downloading a model.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalBackendError, SignalConfigError, SignalDataError
from flexsignal.testing import synthetic

foundation = importlib.import_module("flexsignal.adapters.foundation")
TIMESFM = "google/timesfm-2.5-200m-pytorch"
T0 = "theforecastingcompany/t0-alpha"


def _hourly(days=3, streams=("flow",)):
    case = synthetic.plant(seed=1, periods=days * 24, freq="1h")
    return case.signal().select(list(streams))


def _frames(days=3, exog=False):
    values = _hourly(days, ("flow", "temperature") if exog else ("flow",)).values
    history = values.iloc[:-6]
    future = values.iloc[-6:].drop(columns="flow")
    return history, future


# -- a stand-in timesfm ------------------------------------------------------------


class _FakeTimesFM:
    seen: dict = {}

    @classmethod
    def from_pretrained(cls, model, revision):
        cls.seen.update(model=model, revision=revision)
        return cls()

    def compile(self, config):
        self.config = config

    def forecast(self, horizon, inputs):
        _FakeTimesFM.seen["inputs"] = inputs
        last = inputs[0][-1]
        point = np.full((1, horizon), last)
        # TimesFM returns the mean then q10..q90 along the last axis.
        spread = np.arange(10)[None, None, :] - 5.0
        return point, last + np.broadcast_to(spread, (1, horizon, 10))


@pytest.fixture
def timesfm(monkeypatch):
    module = types.SimpleNamespace(
        TimesFM_2p5_200M_torch=_FakeTimesFM,
        ForecastConfig=lambda **kw: kw,
    )
    monkeypatch.setitem(sys.modules, "timesfm", module)
    _FakeTimesFM.seen = {}
    foundation._timesfm.cache_clear()
    yield module
    foundation._timesfm.cache_clear()


@pytest.mark.unit
def test_timesfm_forecasts_at_the_pinned_revision(timesfm):
    history, future = _frames()
    history.iloc[-3, 0] = np.nan  # gaps are interpolated before TimesFM sees them
    out = foundation.chronos_forecast(history, future, model=TIMESFM)
    assert _FakeTimesFM.seen["revision"] == foundation.PINNED_REVISIONS[TIMESFM]
    assert not np.isnan(_FakeTimesFM.seen["inputs"][0]).any()
    assert set(out) == {"mean", *(f"q{10 * i:02d}" for i in range(1, 10))}
    assert len(out["mean"]) == 6
    assert (out["q10"] < out["q50"]).all() and (out["q50"] < out["q90"]).all()


@pytest.mark.unit
def test_timesfm_refuses_exog(timesfm):
    history, future = _frames(exog=True)
    with pytest.raises(SignalConfigError, match="don't take exog") as err:
        foundation.chronos_forecast(history, future, model=TIMESFM)
    assert err.value.field == "exog"


# -- a stand-in t0 -------------------------------------------------------------------


class _FakeT0:
    seen: dict = {}

    @classmethod
    def from_pretrained(cls, model, revision):
        if revision == "missing":
            raise OSError("revision not found")
        cls.seen.update(model=model, revision=revision)
        return cls()

    def eval(self):
        return self

    def predict(self, context, horizon, quantile_levels, future_covariates=None):
        import torch

        _FakeT0.seen.update(context=context, covariates=future_covariates)
        last = float(context[0, -1])
        levels = torch.tensor(quantile_levels, dtype=torch.float32)
        q = last + (levels - 0.5)[None, None, :].expand(1, horizon, len(levels))
        return types.SimpleNamespace(quantiles=q, median=q[..., len(levels) // 2])


@pytest.fixture
def t0(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "t0", types.SimpleNamespace(T0Forecaster=_FakeT0))
    foundation._t0.cache_clear()
    _FakeT0.seen = {}
    yield
    foundation._t0.cache_clear()


@pytest.mark.unit
def test_t0_passes_exog_as_covariates_over_history_and_horizon(t0):
    history, future = _frames(exog=True)
    out = foundation.t0_forecast(history, future)
    assert _FakeT0.seen["revision"] == foundation.PINNED_REVISIONS[T0]
    covariates = _FakeT0.seen["covariates"]
    assert tuple(covariates.shape) == (1, 1, len(history) + len(future))
    assert set(out) >= {"mean", "q01", "q50", "q99"}
    assert (out["q10"] < out["q90"]).all()
    np.testing.assert_array_equal(out["mean"], out["q50"])


@pytest.mark.unit
def test_t0_runs_as_a_step_without_exog(t0):
    step = fs.forecast.T0Forecast("flow", horizon="6h", quantiles=[0.1, 0.9])
    out = fs.run(fs.Pipeline([step]).spec, _hourly()).signal
    ahead = out.has_flag(fs.Flag.FORECAST)["flow"]
    assert ahead.sum() == 6 and _FakeT0.seen["covariates"] is None
    assert (out.values["flow__q10"][ahead] < out.values["flow__q90"][ahead]).all()


@pytest.mark.unit
def test_t0_says_when_the_weights_cant_be_loaded(t0):
    history, future = _frames()
    with pytest.raises(SignalBackendError, match="could not load"):
        foundation.t0_forecast(history, future, revision="missing")


@pytest.mark.unit
def test_chronos_says_when_the_weights_cant_be_loaded(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no network")

    chronos = types.SimpleNamespace(
        Chronos2Pipeline=types.SimpleNamespace(from_pretrained=refuse)
    )
    monkeypatch.setitem(sys.modules, "chronos", chronos)
    monkeypatch.setattr(foundation, "require", lambda name, *a, **k: chronos)
    foundation._chronos.cache_clear()
    try:
        with pytest.raises(SignalBackendError, match="could not load") as err:
            foundation._chronos("amazon/chronos-2", "abc")
        assert "Check the network connection" in str(err.value)
    finally:
        foundation._chronos.cache_clear()


# -- checks before any model runs ----------------------------------------------------


@pytest.mark.unit
def test_forecast_rows_must_follow_the_history():
    history, future = _frames()
    gap = future.copy()
    gap.index = gap.index + pd.Timedelta("1h")  # one step missing in between
    for fn in (foundation.chronos_forecast, foundation.t0_forecast):
        with pytest.raises(SignalDataError, match="one data step"):
            fn(history, gap)


@pytest.mark.unit
def test_history_needs_two_values_and_revision_must_be_a_string():
    history, future = _frames()
    empty = history.copy()
    empty.iloc[:-1, 0] = np.nan
    with pytest.raises(SignalDataError, match="at least two values"):
        foundation.chronos_forecast(empty, future)
    with pytest.raises(SignalConfigError, match="revision must be") as err:
        foundation.t0_forecast(history, future, revision="")
    assert err.value.field == "revision"
