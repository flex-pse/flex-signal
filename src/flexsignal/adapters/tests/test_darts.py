"""The [darts] adapter: fs.forecast.DartsForecast."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

import flexsignal as fs  # noqa: E402
from flexsignal.core import Flag, SignalBackendError, SignalConfigError  # noqa: E402
from flexsignal.testing import synthetic  # noqa: E402

PLUGIN = "flexsignal.adapters.darts"
NAME = "DartsForecast"


def _plant(days=14, seed=1):
    case = synthetic.plant(freq="1h", periods=days * 24, seed=seed)
    return case.signal().select(["flow", "temperature"])


def _issue(sf, hours_before_end=24):
    return sf.index[-hours_before_end - 1].isoformat()


def _spec(step):
    return fs.Pipeline([step], name="darts_forecast").spec


SPEC = {
    "schema_version": "0.1.0",
    "name": "darts_forecast",
    "requires": [PLUGIN],
    "steps": [
        {
            "kind": NAME,
            "streams": ["flow"],
            "horizon": "1D",
            "quantiles": [0.1, 0.9],
            "model": "ExponentialSmoothing",
            "params": {"trend": None},
            "season": "1D",
            "num_samples": 200,
        }
    ],
}


def _python_step(**extra):
    return fs.forecast.DartsForecast(
        "flow",
        horizon="1D",
        quantiles=[0.1, 0.9],
        model="ExponentialSmoothing",
        params={"trend": None},
        season="1D",
        num_samples=200,
        **extra,
    )


@pytest.mark.component
def test_python_and_json_agree_save_and_repeat(tmp_path):
    sf = _plant()
    from_python = fs.run(_spec(_python_step()), sf)
    from_json = fs.run(json.dumps(SPEC), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    spec = from_json.fitted.spec
    assert fs.load_spec(fs.dump_spec(spec)) == spec

    out = from_json.signal
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 24
    assert str(out.index.tz) == "UTC" and out.index[-1] == sf.index[-1] + pd.Timedelta(
        "1D"
    )
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
        ("model", "darts.models.Prophet"),
        ("model", "darts.models:TFTModel"),
        ("num_samples", "many"),
        ("params", [1, 2]),
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
        ({"params": {"nope": 1}}, "params", "'nope' isn't an argument of darts'"),
        ({"params": {"trend": "sideways"}}, "params", "'trend' of Exponential"),
        ({"params": {"add_encoders": {}}}, "params", "set by flex-signal"),
        ({"num_samples": 5}, "num_samples", "between 10 and 10000"),
        ({"season": "soon"}, "season", "isn't a duration"),
        ({"model": "LinearRegressionModel", "season": None}, "params", "needs lags"),
    ],
)
def test_bad_params_are_refused_by_name(options, field, words):
    step = fs.forecast.DartsForecast("flow", horizon="6h", **options)
    with pytest.raises(SignalConfigError, match=words) as err:
        fs.run(_spec(step), _plant(days=4))
    assert err.value.field == field


@pytest.mark.unit
def test_a_missing_darts_names_the_extra(monkeypatch):
    for module in ("darts", "darts.models", "darts.utils.utils"):
        monkeypatch.setitem(sys.modules, module, None)
    step = fs.forecast.DartsForecast("flow", horizon="6h")
    with pytest.raises(SignalBackendError, match=r"flex-signal\[darts\]"):
        fs.run(_spec(step), _plant(days=4))


@pytest.mark.component
def test_registering_imports_no_darts():
    code = (
        "import sys, flexsignal as fs; fs.load_plugins('flexsignal.adapters.darts'); "
        "from flexsignal.backends import get_backend; "
        "print(fs.forecast.DartsForecast.__name__, 'darts' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == [NAME, "False"]


@pytest.mark.unit
def test_backend_keys_name_the_extra():
    from flexsignal.backends import registered_backends

    keys = {b.key: b for b in registered_backends() if b.key.startswith("darts:")}
    assert "darts:ExponentialSmoothing" in keys and "darts:ARIMA" in keys
    assert {info.extra for info in keys.values()} == {"darts"}


@pytest.mark.integration
def test_a_seasonal_model_beats_persistence_in_a_backtest():
    result = fs.backtest(
        _spec(fs.forecast.DartsForecast("flow", horizon="1D")),
        _plant(),
        folds=3,
        compare=[
            fs.forecast.Persistence("flow", horizon="1D"),
            fs.forecast.DartsForecast(
                "flow", horizon="1D", model="Theta", name="darts_theta"
            ),
            fs.forecast.DartsForecast(
                "flow", horizon="1D", model="NaiveSeasonal", name="darts_naive"
            ),
        ],
        baseline="forecast_persistence",
    )
    summary = result.summary().set_index("method")
    print(summary.to_string())
    assert summary.loc[NAME, "skill"] > 0.5
    assert summary.loc["darts_theta", "skill"] > 0.5
    assert summary.loc["darts_naive", "skill"] > 0.3
    assert len(result.scores) == 12 and (result.scores["n"] == 24).all()


@pytest.mark.component
@pytest.mark.parametrize("model", ["ARIMA", "LinearRegressionModel"])
def test_exog_streams_are_future_covariates(model):
    sf = _plant()
    issue = _issue(sf)
    truth = sf.values["flow"].iloc[-24:]
    errors = {}
    for exog in ((), ("temperature",)):
        step = fs.forecast.DartsForecast(
            "flow",
            horizon="1D",
            issue_time=issue,
            exog=exog,
            model=model,
            season=None if model == "ARIMA" else "1D",
        )
        out = fs.run(_spec(step), sf).signal
        assert out.has_flag(Flag.FORECAST)["flow"].sum() == 24
        predicted = out.values["flow"].iloc[-24:]
        errors[exog] = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    # Without a season, ARIMA learns the daily cycle only from the exog stream.
    assert errors[("temperature",)] < 3.0
    if model == "ARIMA":
        assert errors[("temperature",)] < errors[()] / 3


@pytest.mark.component
def test_models_without_exog_or_intervals_refuse_them_clearly():
    sf = _plant(days=4)
    step = fs.forecast.DartsForecast(
        "flow", horizon="6h", issue_time=_issue(sf), exog="temperature", model="Theta"
    )
    with pytest.raises(SignalConfigError, match="Theta takes no exog") as err:
        fs.run(_spec(step), sf)
    assert err.value.field == "exog"
    step = fs.forecast.DartsForecast(
        "flow", horizon="6h", model="Theta", quantiles=[0.1]
    )
    with pytest.raises(SignalConfigError, match="Theta gives no intervals") as err:
        fs.run(_spec(step), sf)
    assert err.value.field == "quantiles"


@pytest.mark.component
def test_linear_regression_gives_quantiles_with_a_quantile_likelihood():
    step = fs.forecast.DartsForecast(
        "flow",
        horizon="6h",
        quantiles=[0.1, 0.5, 0.9],
        model="LinearRegressionModel",
        params={"likelihood": "quantile", "quantiles": [0.1, 0.5, 0.9]},
    )
    out = fs.run(_spec(step), _plant(days=6)).signal
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    q10, q90 = out.values["flow__q10"][ahead], out.values["flow__q90"][ahead]
    assert (q10 < q90).all()


@pytest.mark.component
def test_gaps_and_a_late_issue_time_are_handled():
    sf = _plant(days=6)
    values = sf.values.copy()
    values.iloc[40:46, 0] = np.nan  # a gap in the history
    values.iloc[-30:, 0] = np.nan  # the target stops 6 h before the issue time
    sf = fs.SignalFrame.from_frame(
        values, units={"flow": "m^3/hr", "temperature": "degC"}
    )
    step = fs.forecast.DartsForecast(
        "flow", horizon="1D", issue_time=_issue(sf), quantiles=[0.1, 0.9]
    )
    out = fs.run(_spec(step), sf).signal
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 24 and out.values["flow"][ahead].notna().all()
    assert out.values["flow"][ahead].between(50, 150).all()
