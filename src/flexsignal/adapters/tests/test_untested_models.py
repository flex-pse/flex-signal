"""Adapters run a library's other models too, by class name, untested.

A spec names a model by bare class name only, never an import path. The tested
models keep flex-signal's simple settings; any other model of the library runs
with its own settings in ``params``, and an error names what it still needs.
"""

import warnings

import numpy as np
import pytest

import flexsignal as fs
from flexsignal.adapters._common import MODEL_OPTIONS
from flexsignal.core import SignalConfigError
from flexsignal.testing import synthetic

STEPS = {
    "PyodDetector": (fs.clean, "model"),
    "RupturesChangepoint": (fs.clean, "algorithm"),
    "PypotsImpute": (fs.impute, "model"),
    "PypotsForecast": (fs.forecast, "model"),
    "NeuralforecastForecast": (fs.forecast, "model"),
    "DartsForecast": (fs.forecast, "model"),
    "SktimeForecast": (fs.forecast, "model"),
}


def _five_minute(days=4):
    return synthetic.plant(seed=1, periods=days * 288).signal().select(["flow"])


def _hourly(days=8):
    case = synthetic.plant(seed=1, periods=days * 24, freq="1h")
    return case.signal().select(["flow"])


def _run(step, sf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fs.run(fs.Pipeline([step]).spec, sf).signal


@pytest.mark.unit
def test_every_model_option_is_recorded_for_the_docs():
    assert {o.method for o in MODEL_OPTIONS} == set(STEPS)
    for option in MODEL_OPTIONS:
        assert option.tested and option.catalogue.startswith("https://")
        group, field = STEPS[option.method]
        assert option.option == field
        assert getattr(group, option.method).model_fields[field].default in (
            option.tested
        )


@pytest.mark.unit
@pytest.mark.parametrize("method", sorted(STEPS))
def test_any_class_name_loads_but_never_an_import_path(method):
    group, field = STEPS[method]
    cls = getattr(group, method)
    extra = {"horizon": "6h"} if group is fs.forecast else {}
    step = cls("flow", **{field: "SomeOtherModel"}, **extra)
    assert getattr(step, field) == "SomeOtherModel"
    for bad in ("os.system", "pyod.models.knn:KNN", "../KNN", "_private", ""):
        with pytest.raises((SignalConfigError, ValueError)):
            cls("flow", **{field: bad}, **extra)


@pytest.mark.integration
def test_pyod_runs_another_detector_with_its_params():
    pytest.importorskip("pyod")
    sf = _five_minute()
    out = _run(fs.clean.PyodDetector("flow", model="PCA"), sf)
    assert out.has_flag(fs.Flag.OUTLIER)["flow"].sum() > 0
    with pytest.raises(SignalConfigError, match="has no model 'NoSuchDetector'"):
        _run(fs.clean.PyodDetector("flow", model="NoSuchDetector"), sf)
    with pytest.raises(SignalConfigError, match="use the step's own options"):
        _run(fs.clean.PyodDetector("flow", params={"contamination": 0.2}), sf)


@pytest.mark.integration
def test_ruptures_runs_another_search_and_refuses_one_without_a_penalty():
    pytest.importorskip("ruptures")
    sf = _five_minute()
    step = fs.clean.RupturesChangepoint(
        "flow", algorithm="Window", params={"width": 20}
    )
    _run(step, sf)
    with pytest.raises(SignalConfigError, match="can't search with a penalty"):
        _run(fs.clean.RupturesChangepoint("flow", algorithm="Dynp"), sf)


@pytest.mark.integration
def test_pypots_runs_another_model_and_names_the_settings_it_needs():
    pytest.importorskip("pypots")
    values = _hourly().values.copy()
    values.iloc[50:56, 0] = np.nan
    sf = fs.SignalFrame.from_frame(values, units={"flow": "m3/h"})
    out = _run(fs.impute.PypotsImpute("flow", model="FreTS", epochs=2), sf)
    assert out.values["flow"].notna().all()
    with pytest.raises(SignalConfigError, match=r"needs \['patch_size'") as err:
        _run(fs.impute.PypotsImpute("flow", model="PatchTST", epochs=2), sf)
    assert "give them in params" in str(err.value)
    step = fs.forecast.PypotsForecast(
        "flow", horizon="6h", model="SegRNN", epochs=2, params={"seg_len": 6}
    )
    assert _run(step, _hourly()).has_flag(fs.Flag.FORECAST)["flow"].sum() == 6


@pytest.mark.integration
def test_neuralforecast_runs_another_model_and_a_multivariate_one():
    pytest.importorskip("neuralforecast")
    for model in ("RNN", "TSMixer"):  # TSMixer needs n_series, set to 1
        step = fs.forecast.NeuralforecastForecast(
            "flow", horizon="6h", model=model, max_steps=5
        )
        assert _run(step, _hourly()).has_flag(fs.Flag.FORECAST)["flow"].sum() == 6


@pytest.mark.integration
def test_darts_runs_another_model():
    pytest.importorskip("darts")
    step = fs.forecast.DartsForecast(
        "flow",
        horizon="6h",
        model="RandomForestModel",
        season=None,
        params={"lags": 24},
    )
    assert _run(step, _hourly()).has_flag(fs.Flag.FORECAST)["flow"].sum() == 6


@pytest.mark.integration
def test_sktime_runs_another_forecaster():
    pytest.importorskip("sktime")
    step = fs.forecast.SktimeForecast(
        "flow", horizon="6h", model="TrendForecaster", season=None
    )
    assert _run(step, _hourly()).has_flag(fs.Flag.FORECAST)["flow"].sum() == 6
