"""Regression tests for the adapter fixes of the 0.1.0 review.

Each test is small and skips when the library it needs isn't installed; the
shared helpers in ``_common`` are tested with stand-in classes.
"""

import abc
import random
import sys
import types

import numpy as np
import pytest

import flexsignal as fs
from flexsignal.adapters import _common
from flexsignal.core import SignalConfigError, SignalDataError
from flexsignal.testing import synthetic


def _values(freq="1h", periods=96, streams=("flow",), seed=1):
    case = synthetic.plant(freq=freq, periods=periods, seed=seed)
    return case.signal().select(list(streams)).values


def _split(values, n=6, exog=()):
    """History, and the forecast rows with only the exog columns."""
    return values.iloc[:-n], values.iloc[-n:][list(exog)]


# -- season: only for the models that use it (finding 1) ---------------------------


@pytest.mark.component
@pytest.mark.parametrize("model", ["ExponentialSmoothing", "ARIMA", "NaiveDrift"])
def test_darts_default_season_on_daily_data_is_no_cycle(model):
    pytest.importorskip("darts")
    darts = pytest.importorskip("flexsignal.adapters.darts")
    history, future = _split(_values(freq="1D", periods=60), n=5)
    out = darts.darts_forecast(history, future, model=model)
    assert np.isfinite(out["mean"]).all() and len(out["mean"]) == 5


@pytest.mark.component
def test_darts_season_is_ignored_by_models_that_dont_use_it():
    pytest.importorskip("darts")
    darts = pytest.importorskip("flexsignal.adapters.darts")
    history, future = _split(_values(freq="7min", periods=200), n=10)
    for model in ("NaiveDrift", "NaiveMean"):
        assert len(darts.darts_forecast(history, future, model=model)["mean"]) == 10
    with pytest.raises(SignalDataError, match=r"season \(default 1D\)") as err:
        darts.darts_forecast(history, future, model="NaiveSeasonal")
    assert "season=null" in str(err.value) and err.value.field == "season"
    with pytest.raises(SignalConfigError, match="isn't a duration"):
        darts.darts_forecast(history, future, model="NaiveDrift", season="soon")


@pytest.mark.component
def test_sktime_default_season_on_daily_and_seven_minute_data():
    pytest.importorskip("sktime")
    sktime = pytest.importorskip("flexsignal.adapters.sktime")
    history, future = _split(_values(freq="1D", periods=60), n=5)
    for model in ("UnobservedComponents", "ExponentialSmoothing"):
        out = sktime.sktime_forecast(history, future, model=model)
        assert np.isfinite(out["mean"]).all()
    history, future = _split(_values(freq="7min", periods=200), n=10)
    out = sktime.sktime_forecast(history, future, model="PolynomialTrendForecaster")
    assert len(out["mean"]) == 10
    with pytest.raises(SignalDataError, match="set season=null") as err:
        sktime.sktime_forecast(history, future, model="ThetaForecaster")
    assert err.value.field == "season"


# -- foundation models: the context window is checked (finding 2) ------------------


@pytest.mark.unit
def test_foundation_models_check_the_context_window_not_the_whole_history():
    foundation = pytest.importorskip("flexsignal.adapters.foundation")
    history, future = _split(_values(periods=72))
    history.iloc[-24:, 0] = np.nan  # values before, none in the last day
    for fn in (foundation.chronos_forecast, foundation.t0_forecast):
        with pytest.raises(SignalDataError, match=r"last 12h \(context\)"):
            fn(history, future, context="12h")


# -- neuralforecast (findings 3 and 5) ---------------------------------------------


@pytest.mark.component
def test_neuralforecast_refuses_a_missing_future_exog_value():
    pytest.importorskip("neuralforecast")
    deep = pytest.importorskip("flexsignal.adapters.deep")
    history, future = _split(
        _values(streams=("flow", "temperature")), exog=["temperature"]
    )
    future.iloc[-2, 0] = np.nan
    with pytest.raises(SignalDataError, match="'temperature' has no value .* forecast"):
        deep.neuralforecast_forecast(history, future, model="NHITS", max_steps=1)


@pytest.mark.component
def test_neuralforecast_names_the_quantiles_it_trained():
    pytest.importorskip("neuralforecast")
    fs.load_plugins("flexsignal.adapters.deep")
    step = fs.forecast.NeuralforecastForecast(
        "flow", horizon="6h", quantiles=[0.25], model="MLP", max_steps=2
    )
    sf = synthetic.plant(freq="1h", periods=72, seed=1).signal().select(["flow"])
    with pytest.raises(SignalConfigError, match="trains the quantiles 0.01") as err:
        fs.run(fs.Pipeline([step]).spec, sf)
    assert "not 0.25" in str(err.value) and err.value.field == "quantiles"


# -- seeding leaves the caller's random state alone (finding 4) --------------------


@pytest.mark.unit
def test_seeded_restores_python_and_numpy_random_state():
    random.seed(7)
    np.random.seed(7)
    with _common.seeded(0):
        inside = random.random(), np.random.rand()
    assert random.random() == random.Random(7).random()
    assert np.random.rand() == np.random.RandomState(7).rand()
    with _common.seeded(0):
        assert (random.random(), np.random.rand()) == inside


@pytest.mark.unit
def test_seeded_restores_torch_rng_and_the_deterministic_flag():
    torch = pytest.importorskip("torch")
    before = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    with _common.seeded(0):
        assert torch.are_deterministic_algorithms_enabled()
        first = torch.rand(3)
    assert torch.are_deterministic_algorithms_enabled() == before
    assert torch.equal(torch.rand(3), expected)
    with _common.seeded(0):
        assert torch.equal(torch.rand(3), first)


@pytest.mark.component
def test_pypots_still_repeats_and_keeps_the_callers_random_state():
    pypots = pytest.importorskip("flexsignal.adapters.pypots")
    pytest.importorskip("pypots")
    history, future = _split(_values(periods=72))
    np.random.seed(123)
    a = pypots.pypots_forecast(history, future, lookback="12h", epochs=1)["mean"]
    assert np.random.rand() == np.random.RandomState(123).rand()
    b = pypots.pypots_forecast(history, future, lookback="12h", epochs=1)["mean"]
    np.testing.assert_array_equal(a, b)


# -- model classes: never a base or abstract class (finding 6) ---------------------


class _Base(abc.ABC):
    @abc.abstractmethod
    def fit(self): ...


class _Abstract(_Base):
    pass


class _Model(_Base):
    def __init__(self, size=1):
        self.size = size

    def fit(self): ...


_OPTION = _common.ModelOption(
    method="Fake", option="model", library="fakelib", tested=("_Model",), catalogue="x"
)


@pytest.mark.unit
def test_model_class_refuses_the_base_and_abstract_classes():
    module = types.SimpleNamespace(Base=_Base, Abstract=_Abstract, Model=_Model)
    assert _common.model_class(module, "Model", _OPTION, base=_Base) is _Model
    for name in ("Base", "Abstract"):
        with pytest.raises(SignalConfigError, match=f"has no model '{name}'") as err:
            _common.model_class(module, name, _OPTION, base=_Base)
        assert err.value.field == "model"


@pytest.mark.unit
def test_pyod_refuses_its_base_detector_and_finds_typos_quickly(capfd):
    anomaly = pytest.importorskip("flexsignal.adapters.anomaly")
    pytest.importorskip("pyod")
    with pytest.raises(SignalConfigError, match="has no model 'BaseDetector'"):
        anomaly._pyod_class("BaseDetector", "PyodDetector")
    with pytest.raises(SignalConfigError, match="has no model 'IForrest'"):
        anomaly._pyod_class("IForrest", "PyodDetector")
    assert "please install" not in capfd.readouterr().out
    assert "pyod.models.xgbod" not in sys.modules
    # An untested detector whose module isn't named after it is still found.
    assert anomaly._pyod_class("SO_GAAL", "PyodDetector").__name__ == "SO_GAAL"


# -- one forecast mapping, one params check (findings 8 and 9) ---------------------


@pytest.mark.unit
def test_forecast_reads_quantiles_lazily_and_says_why_it_has_none():
    calls = []

    def quantile(q):
        calls.append(q)
        return np.full(2, q)

    lazy = _common.Forecast(np.zeros(2), quantile, "never")
    assert "q10" in lazy and calls == []
    np.testing.assert_array_equal(lazy["q10"], [0.1, 0.1])
    lazy["q10"]
    assert calls == [0.1] and list(lazy) == ["mean"] and "std" not in lazy
    point = _common.Forecast(np.zeros(2), None, "no intervals; remove quantiles")
    with pytest.raises(SignalConfigError, match="no intervals") as err:
        "q10" in point  # noqa: B015 - the lookup is what is tested
    assert err.value.field == "quantiles"
    some = _common.Forecast(
        np.zeros(2), {"q50": np.ones(2)}, lambda q: f"has only q50, not {q:g}"
    )
    assert "q50" in some and list(some) == ["mean", "q50"]
    with pytest.raises(SignalConfigError, match="not 0.25"):
        "q25" in some  # noqa: B015


class _Wrapper:
    def __init__(self, size=1, **kwargs):
        self.size, self.kwargs = size, kwargs


@pytest.mark.unit
def test_build_model_hooks():
    option = _common.ModelOption(
        method="Fake", option="model", library="fakelib", tested=(), catalogue="x"
    )
    loose = _common.build_model(_Wrapper, {"size": 2}, {"anything": 1}, option)
    assert loose.kwargs == {"anything": 1}
    strict = {"strict": True, "extra_accepted": ["passed_on"]}
    model = _common.build_model(_Wrapper, {}, {"passed_on": 1}, option, **strict)
    assert model.kwargs == {"passed_on": 1}
    with pytest.raises(SignalConfigError, match="'anything' isn't an argument"):
        _common.build_model(_Wrapper, {}, {"anything": 1}, option, **strict)
    with pytest.raises(SignalConfigError, match="can't set 'size': why") as err:
        _common.build_model(
            _Wrapper, {}, {"size": 3}, option, managed={"size"}, why_managed="why"
        )
    assert err.value.field == "params" and err.value.value == ["size"]
    with pytest.raises(SignalConfigError, match="params must be a dict"):
        _common.build_model(_Wrapper, {}, ["size"], option)


@pytest.mark.unit
def test_quantile_key():
    assert [_common.quantile_key(q) for q in (0.01, 0.1, 0.5, 0.99)] == [
        "q01",
        "q10",
        "q50",
        "q99",
    ]


# -- TimesFM's loader is cached (finding 13) ---------------------------------------


@pytest.mark.unit
def test_timesfm_loads_once(monkeypatch):
    foundation = pytest.importorskip("flexsignal.adapters.foundation")
    loads = []

    class FakeTimesFM:
        @classmethod
        def from_pretrained(cls, model, revision):
            loads.append(revision)
            return cls()

        def compile(self, config):
            pass

        def forecast(self, horizon, inputs):
            return np.zeros((1, horizon)), np.zeros((1, horizon, 10))

    module = types.SimpleNamespace(
        TimesFM_2p5_200M_torch=FakeTimesFM, ForecastConfig=lambda **kw: kw
    )
    monkeypatch.setitem(sys.modules, "timesfm", module)
    foundation._timesfm.cache_clear()
    try:
        history, future = _split(_values(periods=48))
        for _ in range(2):
            foundation.chronos_forecast(
                history, future, model="google/timesfm-2.5-200m-pytorch"
            )
        assert len(loads) == 1
    finally:
        foundation._timesfm.cache_clear()


# -- ruptures KernelCPD takes a kernel, not a cost (finding 14) --------------------


@pytest.mark.unit
def test_kernel_cpd_refuses_a_cost_it_cant_use():
    anomaly = pytest.importorskip("flexsignal.adapters.anomaly")
    pytest.importorskip("ruptures")
    values = _values(freq="5min", periods=200)["flow"]
    assert len(anomaly.ruptures_changepoint(values, algorithm="KernelCPD")) == 200
    with pytest.raises(SignalConfigError, match="takes a kernel, not a cost") as err:
        anomaly.ruptures_changepoint(values, algorithm="KernelCPD", cost="l1")
    assert err.value.field == "cost"
