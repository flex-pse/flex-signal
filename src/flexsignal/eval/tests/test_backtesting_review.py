"""Backtest fixes from the 0.1.0 code review."""

import json

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core.registry import unregister_step
from flexsignal.eval import metrics
from flexsignal.testing import synthetic


def _data(days=4):
    return synthetic.plant(seed=2, periods=days * 288).signal()


@pytest.mark.component
def test_a_spec_path_resolves_relative_inputs_like_run(tmp_path, monkeypatch):
    sub = tmp_path / "project"
    sub.mkdir()
    fs.io.write_parquet(_data(), sub / "data.parquet")
    spec = fs.Pipeline(
        [fs.forecast.Persistence("flow", horizon="1h")],
        name="bt",
        inputs=[{"kind": "parquet", "name": "plant", "path": "data.parquet"}],
    ).spec
    fs.dump_spec(spec, sub / "spec.json")
    monkeypatch.chdir(tmp_path)  # not the spec's directory
    fs.run(sub / "spec.json", write_outputs=False)
    result = fs.backtest(sub / "spec.json", folds=1)
    assert result.scores["n"].iloc[0] > 0


@pytest.mark.component
def test_a_compared_method_keeps_its_own_exog():
    spec = fs.Pipeline([fs.forecast.Persistence("flow", horizon="1h")], name="bt").spec
    other = fs.forecast.Regression(
        "flow", horizon="1h", fourier=[{"period": "1D"}], exog=["temperature"]
    )
    result = fs.backtest(spec, _data(), folds=2, compare=[other])
    assert set(result.scores["method"]) == {
        "forecast_persistence",
        "forecast_regression",
    }


@pytest.mark.unit
def test_metrics_skip_missing_pairs_and_say_nothing_when_none_are_left():
    found = metrics(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, np.nan, 5.0]))
    assert found["rmse"] == pytest.approx(np.sqrt(2.0))
    empty = metrics(pd.Series([1.0]), pd.Series([np.nan]))
    assert all(np.isnan(v) for v in empty.values())


@pytest.mark.component
def test_forecast_holes_count_as_missing_not_as_scored():
    def holes(history, future):
        out = np.full(len(future), float(history["flow"].dropna().iloc[-1]))
        out[::2] = np.nan
        return out

    fs.register_forecaster("HoleyForecast", holes)
    try:
        spec = fs.Pipeline(
            [fs.forecast.HoleyForecast("flow", horizon="1h")], name="h"
        ).spec
        scores = fs.backtest(spec, _data(), folds=1).scores
        assert scores["n"].iloc[0] == 6 and scores["missing"].iloc[0] == 6
        assert json.loads(json.dumps(fs.backtest(spec, _data(), folds=1).to_dict()))
    finally:
        unregister_step("HoleyForecast")
