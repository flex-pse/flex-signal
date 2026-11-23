"""Backtests: folds, no peeking at the future, scores against baselines."""

import json

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError, SignalDataError
from flexsignal.testing import synthetic

SEASONS = [{"period": "1D", "harmonics": 2}, {"period": "7D", "harmonics": 1}]


def _plant(days=14, seed=1):
    case = synthetic.plant(seed=seed, periods=days * 288).with_spikes("flow", count=8)
    return case.signal().select(["flow", "temperature"])


def _spec(forecaster):
    pipe = fs.Pipeline(name="flow_forecast")
    pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6))
    pipe.add_step(fs.select.Resample(freq="1h"))
    pipe.add_step(forecaster)
    return pipe.spec


@pytest.mark.component
def test_structural_and_regression_beat_seasonal_naive_in_a_backtest():
    """On known daily and weekly cycles, models beat the baseline."""
    result = fs.backtest(
        _spec(fs.forecast.Structural("flow", horizon="1D", seasonal=SEASONS)),
        _plant(),
        folds=3,
        compare=[
            fs.forecast.SeasonalNaive("flow", horizon="1D"),
            fs.forecast.Regression("flow", horizon="1D", fourier=SEASONS),
            fs.forecast.Persistence("flow", horizon="1D"),
        ],
    )
    assert result.baseline == "forecast_seasonal_naive"
    summary = result.summary().set_index("method")
    assert summary.loc["forecast_seasonal_naive", "skill"] == 0.0
    assert summary.loc["forecast_structural", "skill"] > 0.5
    assert summary.loc["forecast_regression", "skill"] > 0.5
    assert summary.loc["forecast_persistence", "skill"] < 0
    # Three folds of 24 hourly values for each of four methods.
    assert len(result.scores) == 12 and (result.scores["n"] == 24).all()
    assert len(result.forecasts) == 4 * 3 * 24


@pytest.mark.component
def test_folds_are_blocked_and_end_with_the_data():
    data = _plant()
    result = fs.backtest(
        _spec(fs.forecast.SeasonalNaive("flow", horizon="12h", quantiles=[0.1, 0.9])),
        data,
        folds=3,
        every="1D",
    )
    issues = sorted(result.scores["issue_time"].unique())
    # The data's last hourly bin starts at 23:00 on the 19th.
    assert issues == [
        pd.Timestamp("2025-01-17T11:00Z"),
        pd.Timestamp("2025-01-18T11:00Z"),
        pd.Timestamp("2025-01-19T11:00Z"),
    ]
    leads = result.forecasts.groupby("fold")["lead"].agg(["min", "max"])
    assert (leads["min"] == pd.Timedelta("1h")).all()
    assert (leads["max"] == pd.Timedelta("12h")).all()
    assert {"q10", "q90"} <= set(result.forecasts.columns)
    assert result.scores["coverage"].between(0, 1).all()
    with pytest.raises(SignalDataError, match="need data from before"):
        fs.backtest(
            _spec(fs.forecast.Persistence("flow", horizon="1D")), data, folds=20
        )


@pytest.mark.component
def test_a_forecast_cant_see_the_data_it_is_scored_on():
    data = _plant(days=10)
    spec = _spec(fs.forecast.Regression("flow", horizon="1D", fourier=SEASONS))
    first = fs.backtest(spec, data, folds=1)
    issue = first.scores["issue_time"].iloc[0]
    # Change every flow value after the issue time: the forecast must not move.
    changed = data.values
    after = changed.index > issue
    changed.loc[after, "flow"] = changed.loc[after, "flow"] * 3.0
    second = fs.backtest(
        spec,
        fs.SignalFrame(changed, data.flags, data.meta),
        folds=1,
    )
    assert second.forecasts["forecast"].equals(first.forecasts["forecast"])
    assert not second.forecasts["actual"].equals(first.forecasts["actual"])


@pytest.mark.component
def test_exog_streams_stay_visible_after_the_issue_time():
    rng = np.random.default_rng(3)
    index = pd.date_range("2025-01-01", periods=24 * 12, freq="1h", tz="UTC")
    price = 0.1 + 0.05 * rng.random(len(index))
    load = 50 - 200 * price + rng.normal(0, 0.1, len(index))
    data = fs.SignalFrame.from_frame(
        pd.DataFrame({"load": load, "price": price}, index=index)
    )
    spec = fs.Pipeline(
        [fs.forecast.Regression("load", horizon="1D", exog=["price"])], name="load"
    ).spec
    result = fs.backtest(spec, data, folds=2)
    assert (result.scores["rmse"] < 0.5).all()


@pytest.mark.component
def test_backtests_are_repeatable_and_json_able():
    data = _plant(days=10)
    spec = _spec(fs.forecast.Structural("flow", horizon="1D", seasonal=SEASONS[:1]))
    compare = [fs.forecast.SeasonalNaive("flow", horizon="1D")]
    first = fs.backtest(spec, data, folds=2, compare=compare)
    second = fs.backtest(fs.dump_spec(spec), data, folds=2, compare=compare)
    assert first.scores.equals(second.scores)
    assert first.forecasts.equals(second.forecasts)
    payload = json.loads(json.dumps(first.to_dict()))
    assert payload["baseline"] == "forecast_seasonal_naive"
    assert payload["scores"][0]["issue_time"].startswith("2025-01-")
    assert {row["method"] for row in payload["summary"]} == {
        "forecast_structural",
        "forecast_seasonal_naive",
    }


@pytest.mark.unit
def test_bad_backtests_are_refused():
    data = _plant(days=4)
    spec = _spec(fs.forecast.Persistence("flow", horizon="1D"))
    no_forecast = fs.Pipeline([fs.select.Resample(freq="1h")], name="p").spec
    with pytest.raises(SignalConfigError, match="exactly one forecast step"):
        fs.backtest(no_forecast, data)
    with pytest.raises(SignalConfigError, match="it must match"):
        fs.backtest(spec, data, compare=[fs.forecast.Persistence("flow", horizon="2h")])
    with pytest.raises(SignalConfigError, match="share a name"):
        fs.backtest(spec, data, compare=[fs.forecast.Persistence("flow", horizon="1D")])
    with pytest.raises(SignalConfigError, match="isn't one of the methods"):
        fs.backtest(spec, data, baseline="nope")
    with pytest.raises(SignalConfigError, match="positive duration"):
        fs.backtest(spec, data, every="soon")
    with pytest.raises(SignalConfigError, match="forecast steps, not"):
        fs.backtest(spec, data, compare=[fs.select.Resample(freq="1h")])
