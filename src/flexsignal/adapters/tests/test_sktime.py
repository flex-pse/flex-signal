"""The [sktime] adapter: fs.forecast.SktimeForecast."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sktime")

import flexsignal as fs  # noqa: E402
from flexsignal.core import Flag, SignalBackendError, SignalConfigError  # noqa: E402
from flexsignal.testing import synthetic  # noqa: E402

PLUGIN = "flexsignal.adapters.sktime"
NAME = "SktimeForecast"


def _plant(days=14, seed=1):
    case = synthetic.plant(freq="1h", periods=days * 24, seed=seed)
    return case.signal().select(["flow", "temperature"])


def _issue(sf, hours_before_end=24):
    return sf.index[-hours_before_end - 1].isoformat()


def _spec(step):
    return fs.Pipeline([step], name="sktime_forecast").spec


SPEC = {
    "schema_version": "0.1.0",
    "name": "sktime_forecast",
    "requires": [PLUGIN],
    "steps": [
        {
            "kind": NAME,
            "streams": ["flow"],
            "horizon": "1D",
            "quantiles": [0.1, 0.9],
            "model": "AutoETS",
            "params": {"error": "add"},
            "season": "1D",
        }
    ],
}


@pytest.mark.component
def test_python_and_json_agree_save_and_repeat(tmp_path):
    sf = _plant()
    step = fs.forecast.SktimeForecast(
        "flow",
        horizon="1D",
        quantiles=[0.1, 0.9],
        model="AutoETS",
        params={"error": "add"},
        season="1D",
    )
    from_python = fs.run(_spec(step), sf)
    from_json = fs.run(json.dumps(SPEC), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    spec = from_json.fitted.spec
    assert fs.load_spec(fs.dump_spec(spec)) == spec

    out = from_json.signal
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 24
    assert str(out.index.tz) == "UTC"
    assert out.index[-1] == sf.index[-1] + pd.Timedelta("1D")
    mean = out.values["flow"][ahead]
    assert (out.values["flow__q10"][ahead] < mean).all()
    assert (out.values["flow__q90"][ahead] > mean).all()

    again = fs.run(json.dumps(SPEC), sf)
    pd.testing.assert_frame_equal(again.data, from_json.data)

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved, plugins=[PLUGIN])
    pd.testing.assert_frame_equal(loaded.transform(sf).values, out.values)


@pytest.mark.unit
@pytest.mark.parametrize(
    "option, value",
    [
        ("model", "sktime.forecasting.arima:AutoARIMA"),
        ("model", "sktime.forecasting.arima:ARIMA"),
        ("params", "sp=24"),
        ("season", 24),
        ("seed", "x"),
    ],
)
def test_bad_options_fail_on_load_with_the_field(option, value):
    spec = json.loads(json.dumps(SPEC))
    spec["steps"][0][option] = value
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(spec)
    assert err.value.field == f"steps.0.{option}"


@pytest.mark.component
@pytest.mark.parametrize(
    "options, field, words",
    [
        ({"params": {"nope": 1}}, "params", "'nope' isn't an argument of sktime's"),
        (
            {"model": "STLForecaster", "params": {"forecaster_trend": {}}},
            "params",
            "takes a Python object",
        ),
        ({"season": "soon"}, "season", "isn't a duration"),
    ],
)
def test_bad_params_are_refused_by_name(options, field, words):
    step = fs.forecast.SktimeForecast("flow", horizon="6h", **options)
    with pytest.raises(SignalConfigError, match=words) as err:
        fs.run(_spec(step), _plant(days=4))
    assert err.value.field == field


@pytest.mark.unit
def test_a_missing_sktime_names_the_extra(monkeypatch):
    for module in [m for m in sys.modules if m == "sktime" or m.startswith("sktime.")]:
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setitem(sys.modules, "sktime", None)
    step = fs.forecast.SktimeForecast("flow", horizon="6h")
    with pytest.raises(SignalBackendError, match=r"flex-signal\[sktime\]"):
        fs.run(_spec(step), _plant(days=4))


@pytest.mark.component
def test_registering_imports_no_sktime():
    code = (
        "import sys, flexsignal as fs; fs.load_plugins('flexsignal.adapters.sktime'); "
        "print(fs.forecast.SktimeForecast.__name__, 'sktime' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == [NAME, "False"]


@pytest.mark.unit
def test_backend_keys_name_the_extra():
    from flexsignal.backends import registered_backends

    keys = {b.key: b for b in registered_backends() if b.key.startswith("sktime:")}
    assert "sktime:ThetaForecaster" in keys and "sktime:SARIMAX" in keys
    assert {info.extra for info in keys.values()} == {"sktime"}


@pytest.mark.integration
def test_a_seasonal_model_beats_persistence_in_a_backtest():
    result = fs.backtest(
        _spec(fs.forecast.SktimeForecast("flow", horizon="1D", quantiles=[0.1, 0.9])),
        _plant(),
        folds=3,
        compare=[
            fs.forecast.Persistence("flow", horizon="1D"),
            fs.forecast.SktimeForecast(
                "flow", horizon="1D", model="UnobservedComponents", name="sktime_uc"
            ),
            fs.forecast.SktimeForecast(
                "flow", horizon="1D", model="NaiveForecaster", name="sktime_naive"
            ),
        ],
        baseline="forecast_persistence",
    )
    summary = result.summary().set_index("method")
    print(summary.to_string())
    assert summary.loc[NAME, "skill"] > 0.5
    assert summary.loc["sktime_uc", "skill"] > 0.5
    assert summary.loc["sktime_naive", "skill"] > 0.3
    assert len(result.scores) == 12 and (result.scores["n"] == 24).all()


@pytest.mark.component
@pytest.mark.parametrize("model", ["SARIMAX", "UnobservedComponents"])
def test_exog_streams_are_X(model):
    sf = _plant()
    issue = _issue(sf)
    truth = sf.values["flow"].iloc[-24:]
    errors = {}
    for exog in ((), ("temperature",)):
        step = fs.forecast.SktimeForecast(
            "flow",
            horizon="1D",
            issue_time=issue,
            exog=exog,
            model=model,
            season=None,
            quantiles=[0.1, 0.9],
        )
        out = fs.run(_spec(step), sf).signal
        assert out.has_flag(Flag.FORECAST)["flow"].sum() == 24
        predicted = out.values["flow"].iloc[-24:]
        errors[exog] = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    # Without a season, the model learns the daily cycle only from the exog stream.
    assert errors[("temperature",)] < 3.0
    assert errors[("temperature",)] < errors[()] / 3


@pytest.mark.component
def test_models_without_exog_or_intervals_refuse_them_clearly():
    sf = _plant(days=4)
    step = fs.forecast.SktimeForecast(
        "flow", horizon="6h", issue_time=_issue(sf), exog="temperature"
    )
    with pytest.raises(SignalConfigError, match="ThetaForecaster takes no exog") as err:
        fs.run(_spec(step), sf)
    assert err.value.field == "exog"
    step = fs.forecast.SktimeForecast(
        "flow", horizon="6h", model="ExponentialSmoothing", quantiles=[0.1]
    )
    with pytest.raises(SignalConfigError, match="gives no intervals") as err:
        fs.run(_spec(step), sf)
    assert err.value.field == "quantiles"


@pytest.mark.component
@pytest.mark.parametrize(
    "model", ["ExponentialSmoothing", "STLForecaster", "PolynomialTrendForecaster"]
)
def test_the_other_models_forecast(model):
    sf = _plant(days=6)
    out = fs.run(
        _spec(fs.forecast.SktimeForecast("flow", horizon="6h", model=model)), sf
    )
    ahead = out.signal.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 6
    assert out.signal.values["flow"][ahead].between(40, 160).all()


@pytest.mark.component
def test_gaps_and_a_late_issue_time_are_handled():
    sf = _plant(days=6)
    values = sf.values.copy()
    values.iloc[40:46, 0] = np.nan  # a gap in the history
    values.iloc[-30:, 0] = np.nan  # the target stops 6 h before the issue time
    sf = fs.SignalFrame.from_frame(
        values, units={"flow": "m^3/hr", "temperature": "degC"}
    )
    step = fs.forecast.SktimeForecast(
        "flow", horizon="1D", issue_time=_issue(sf), quantiles=[0.1, 0.9]
    )
    out = fs.run(_spec(step), sf).signal
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 24 and out.values["flow"][ahead].notna().all()
    assert out.values["flow"][ahead].between(50, 150).all()
