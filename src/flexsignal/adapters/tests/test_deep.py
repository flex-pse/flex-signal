"""The [deep] adapter: NeuralforecastForecast, trained at transform time."""

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.backends import registered_backends
from flexsignal.core import SignalBackendError, SignalConfigError, SignalDataError
from flexsignal.testing import synthetic

deep = importlib.import_module("flexsignal.adapters.deep")
FAST = {"max_steps": 30, "input_size": "1D"}


def _hourly(days=8, seed=1):
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


def _spec(**options):
    step = {"kind": "NeuralforecastForecast", "streams": ["flow"], "horizon": "6h"}
    return {
        "schema_version": "0.1.0",
        "name": "deep",
        "steps": [{**step, **options}],
    }


@pytest.fixture
def neuralforecast():
    return pytest.importorskip("neuralforecast")


# -- registration and options (no training) -----------------------------------------


@pytest.mark.unit
def test_registered_with_documented_options():
    step = fs.forecast.NeuralforecastForecast("flow", horizon="6h")
    assert (step.model, step.input_size, step.max_steps, step.seed) == (
        "NHITS",
        None,
        300,
        0,
    )
    assert (step.loss, step.scaler, step.params) == ("quantile", "standard", None)
    fields = type(step).model_fields
    assert "twice the horizon" in fields["input_size"].description
    keys = {info.key for info in registered_backends()}
    assert {f"neuralforecast:{m}" for m in deep.MODELS} <= keys
    spec = fs.load_spec(_spec(model="MLP", params={"learning_rate": 0.01}, **FAST))
    assert fs.load_spec(fs.dump_spec(spec)) == spec


@pytest.mark.unit
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("model", "neuralforecast.models.TFT"),
        ("max_steps", "lots"),
        ("loss", "huber"),
        ("scaler", "log"),
        ("params", [1]),
        ("input_size", 24),
    ],
)
def test_bad_options_fail_when_the_spec_loads(option, value):
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(_spec(**{option: value}))
    assert err.value.field == f"steps.0.{option}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "field", "match"),
    [
        ({"max_steps": 0}, "max_steps", "at least 1"),
        ({"params": {"h": 3}}, "params.h", "flex-signal sets it"),
        ({"params": {"n_blokcs": [1]}}, "params.n_blokcs", "has no argument"),
    ],
)
def test_bad_values_are_refused_before_training(neuralforecast, options, field, match):
    with pytest.raises(SignalConfigError, match=match) as err:
        fs.run(_spec(**options), _hourly(days=3))
    assert err.value.field == field


@pytest.mark.unit
def test_exog_needs_a_model_that_uses_it(neuralforecast):
    sf, _ = _price_load()
    spec = _spec(model="NBEATS", exog=["price"], horizon="1D")
    spec["steps"][0]["streams"] = ["load"]
    with pytest.raises(SignalConfigError, match="can't use exog") as err:
        fs.run(spec, sf)
    assert err.value.field == "model"


@pytest.mark.unit
def test_too_little_history_names_input_size(neuralforecast):
    with pytest.raises(SignalDataError, match="shorter input_size"):
        fs.run(_spec(input_size="3D"), _hourly(days=3))


@pytest.mark.unit
def test_a_missing_neuralforecast_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "neuralforecast", None)
    with pytest.raises(SignalBackendError, match=r"flex-signal\[deep\]"):
        fs.run(_spec(), _hourly(days=3))


# -- training -----------------------------------------------------------------------


@pytest.mark.integration
def test_runs_from_python_and_json_identically_and_saves(neuralforecast, tmp_path):
    sf = _hourly()
    values = sf.values.copy()
    values.iloc[40:45, 0] = np.nan  # a gap in the history is masked, not trained on
    sf = fs.SignalFrame.from_frame(values, units=sf.units)
    step = fs.forecast.NeuralforecastForecast(
        "flow", horizon="6h", quantiles=[0.1, 0.9], **FAST
    )
    from_python = fs.run(fs.Pipeline([step], name="deep").spec, sf)
    spec = _spec(quantiles=[0.1, 0.9], **FAST)
    from_json = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    again = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(again.data, from_json.data)

    out = from_json.signal
    ahead = out.has_flag(fs.Flag.FORECAST)["flow"]
    assert ahead.sum() == 6
    lo, mid, hi = (out.values[c][ahead] for c in ("flow__q10", "flow", "flow__q90"))
    assert (lo <= hi).all() and (hi > lo).any()
    assert 40 < mid.mean() < 160  # the plant's flow is around 100

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved)
    pd.testing.assert_frame_equal(loaded.transform(sf).values, out.values)


@pytest.mark.integration
def test_point_loss_and_exog(neuralforecast):
    sf, truth = _price_load()
    step = fs.forecast.NeuralforecastForecast(
        "load",
        horizon="1D",
        exog=["price"],
        model="MLP",
        loss="mae",
        max_steps=200,
        params={"learning_rate": 0.01},
    )
    result = fs.run(fs.Pipeline([step]).spec, sf).signal
    error = np.abs(result.values["load"].to_numpy()[-24:] - truth).mean()
    # The price is random, so without it the error would be about 2.5 kW.
    assert error < 1.5, error
    with pytest.raises(SignalDataError, match="no 'std'"):
        fs.run(
            fs.Pipeline(
                [
                    fs.forecast.NeuralforecastForecast(
                        "load", horizon="1D", quantiles=[0.1], loss="mse", **FAST
                    )
                ]
            ).spec,
            sf,
        )


@pytest.mark.integration
def test_beats_persistence_on_daily_cycles(neuralforecast):
    sf = _hourly(days=12)
    step = fs.forecast.NeuralforecastForecast(
        "flow", horizon="1D", quantiles=[0.1, 0.9], input_size="2D", max_steps=100
    )
    result = fs.backtest(
        fs.Pipeline([step]).spec,
        sf,
        folds=2,
        compare=[
            fs.forecast.Persistence("flow", horizon="1D"),
            fs.forecast.SeasonalNaive("flow", horizon="1D"),
        ],
    )
    summary = result.summary().set_index("method")
    print(summary)
    assert summary.loc["NeuralforecastForecast", "skill"] > 0.3
