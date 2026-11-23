"""Forecast steps: where the forecast goes, what it is, and what they refuse."""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError, SignalDataError
from flexsignal.testing import synthetic

SEASONS = [{"period": "1D", "harmonics": 2}, {"period": "7D", "harmonics": 1}]


def _hourly(days=14, seed=1):
    """An hourly plant with daily and weekly cycles, and its noise-free truth."""
    case = synthetic.plant(seed=seed, periods=days * 288)
    pipe = fs.Pipeline([fs.select.Resample(freq="1h")], name="hourly")
    hourly = fs.run(pipe.spec, case.signal()).signal.select(["flow", "temperature"])
    return hourly, case.underlying["flow"].resample("1h").mean()


def _series(values, start="2025-01-01", freq="1h", name="y", units="m^3/hr"):
    index = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    df = pd.DataFrame({name: values}, index=index)
    return fs.SignalFrame.from_frame(df, units={name: units})


def _forecast(step, sf):
    return fs.run(fs.Pipeline([step], name="forecast").spec, sf)


def _rmse(sf, truth):
    rows = sf.has_flag(Flag.FORECAST)["flow"]
    predicted = sf.values["flow"][rows]
    return float(np.sqrt(np.mean((predicted - truth.reindex(predicted.index)) ** 2)))


# -- where the forecast goes ---------------------------------------------------------


@pytest.mark.unit
def test_the_forecast_fills_the_horizon_after_the_issue_time():
    values = np.arange(48.0) + np.random.default_rng(0).normal(0, 0.3, 48)
    values[-1] = 47.0
    sf = _series(values)
    out = _forecast(
        fs.forecast.Persistence("y", horizon="6h", quantiles=[0.1, 0.9]), sf
    )
    result = out.signal
    assert len(result) == 54  # extended by six hourly rows
    forecast = result.has_flag(Flag.FORECAST)["y"]
    assert list(result.index[forecast]) == list(
        pd.date_range("2025-01-03", periods=6, freq="1h", tz="UTC")
    )
    assert (result.values["y"][forecast] == 47.0).all()
    assert result.meta["y"].issue_time == "2025-01-02T23:00:00+00:00"
    # Quantile streams: same units, values only in the forecast rows.
    assert result.columns == ["y", "y__q10", "y__q90"]
    assert result.units["y__q10"] == "m^3/hr"
    low, high = result.values["y__q10"], result.values["y__q90"]
    assert low[~forecast].isna().all() and (low[forecast] < 47).all()
    assert (high[forecast] > 47).all()
    assert result.has_flag(Flag.NO_SAMPLE)["y__q10"][~forecast].all()
    # Persistence intervals widen with the square root of the lead time.
    width = (high - low)[forecast].to_numpy()
    assert width / width[0] == pytest.approx(np.sqrt(np.arange(1, 7)))
    assert out.provenance[-1]["rows_after"] == 54


@pytest.mark.unit
def test_target_values_after_the_issue_time_are_not_used():
    values = np.arange(48.0)
    step = fs.forecast.Persistence("y", horizon="3h", issue_time="2025-01-02T10:00:00Z")
    first = _forecast(step, _series(values)).signal
    later = values.copy()
    later[35:] = -100.0  # after the issue time (index 34)
    second = _forecast(step, _series(later)).signal
    rows = first.has_flag(Flag.FORECAST)["y"]
    assert (first.values["y"][rows] == 34.0).all()
    assert first.values["y"][rows].equals(second.values["y"][rows])
    # Rows after the horizon keep their values; the data isn't extended.
    assert len(first) == 48 and first.values["y"].iloc[-1] == 47.0


@pytest.mark.unit
def test_the_default_issue_time_is_the_last_target_value():
    values = np.arange(48.0)
    values[-5:] = np.nan
    result = _forecast(fs.forecast.Persistence("y", horizon="2h"), _series(values))
    assert result.signal.meta["y"].issue_time == "2025-01-02T18:00:00+00:00"


@pytest.mark.unit
def test_seasonal_naive_repeats_the_last_season_and_skips_missing_values():
    day = np.sin(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    values = np.concatenate([day + 10, day + 20, day + 30])
    values[-24 + 3] = np.nan  # 03:00 on the last day is missing
    step = fs.forecast.SeasonalNaive("y", horizon="1D", season="1D", quantiles=[0.1])
    result = _forecast(step, _series(values)).signal
    rows = result.has_flag(Flag.FORECAST)["y"]
    forecast = result.values["y"][rows].to_numpy()
    expected = day + 30
    expected[3] = day[3] + 20  # one season further back
    assert forecast == pytest.approx(expected)
    spread = forecast - result.values["y__q10"][rows].to_numpy()
    assert spread[3] == pytest.approx(spread[0] * np.sqrt(2))


# -- accuracy ----------------------------------------------------------------------


@pytest.mark.component
def test_structural_and_regression_beat_seasonal_naive_on_a_seasonal_plant():
    hourly, truth = _hourly()
    issue = "2025-01-18T23:00:00Z"
    scores = {
        step.kind: _rmse(_forecast(step, hourly).signal, truth)
        for step in [
            fs.forecast.SeasonalNaive("flow", horizon="1D", issue_time=issue),
            fs.forecast.Structural(
                "flow", horizon="1D", issue_time=issue, seasonal=SEASONS
            ),
            fs.forecast.Regression(
                "flow", horizon="1D", issue_time=issue, fourier=SEASONS
            ),
            fs.forecast.Regression(
                "flow",
                horizon="1D",
                issue_time=issue,
                fourier=[{"period": "7D", "harmonics": 1}],
                lags=["1h", "1D"],
                name="with_lags",
            ),
        ]
    }
    baseline = scores.pop("forecast_seasonal_naive")
    assert baseline > 2.0
    assert all(score < baseline / 2 for score in scores.values()), scores


@pytest.mark.component
def test_structural_intervals_cover_the_truth():
    hourly, truth = _hourly()
    step = fs.forecast.Structural(
        "flow",
        horizon="1D",
        issue_time="2025-01-18T23:00:00Z",
        seasonal=SEASONS,
        quantiles=[0.05, 0.5, 0.95],
    )
    run = _forecast(step, hourly)
    result = run.signal
    rows = result.has_flag(Flag.FORECAST)["flow"]
    low, mid, high = (result.values[f"flow__q{q}"][rows] for q in ("05", "50", "95"))
    actual = truth.reindex(low.index)
    assert ((actual >= low) & (actual <= high)).mean() >= 0.75
    assert mid.to_numpy() == pytest.approx(result.values["flow"][rows].to_numpy())
    metrics = run.fitted.fit_report["steps"][0]["fit_metrics"]["flow"]
    assert metrics["converged"] is True


@pytest.mark.component
def test_exog_streams_known_ahead_drive_the_forecast():
    rng = np.random.default_rng(3)
    index = pd.date_range("2025-01-01", periods=24 * 10, freq="1h", tz="UTC")
    price = 0.1 + 0.05 * rng.random(len(index))
    load = 50 - 200 * price + rng.normal(0, 0.1, len(index))
    load[-24:] = np.nan  # tomorrow's load is unknown; tomorrow's price is known
    sf = fs.SignalFrame.from_frame(
        pd.DataFrame({"load": load, "price": price}, index=index),
        units={"load": "kW", "price": "USD/kWh"},
    )
    truth = 50 - 200 * price[-24:]
    for step in [
        fs.forecast.Regression("load", horizon="1D", exog=["price"]),
        fs.forecast.Structural("load", horizon="1D", exog="price"),
    ]:
        result = _forecast(step, sf).signal
        forecast = result.values["load"].to_numpy()[-24:]
        assert np.abs(forecast - truth).max() < 0.5, step.kind
    # A price missing within the horizon is an error that names it.
    gappy = sf.values.copy()
    gappy.iloc[-5, 1] = np.nan
    with pytest.raises(SignalDataError, match="'price' has no value at 2025-01-10"):
        _forecast(
            fs.forecast.Regression("load", horizon="1D", exog=["price"]),
            fs.SignalFrame.from_frame(gappy),
        )


@pytest.mark.component
def test_past_streams_are_used_at_least_a_horizon_earlier():
    index = pd.date_range("2025-01-01", periods=24 * 6, freq="1h", tz="UTC")
    driver = np.sin(np.arange(len(index)) / 5.0)
    y = np.roll(driver, 6) * 10  # y follows the driver six hours later
    df = pd.DataFrame({"y": y, "driver": driver}, index=index)
    df.loc[df.index[-6:], ["y", "driver"]] = np.nan
    step = fs.forecast.Regression("y", horizon="6h", past=["driver"], past_lags=["6h"])
    result = _forecast(step, fs.SignalFrame.from_frame(df)).signal
    forecast = result.values["y"].to_numpy()[-6:]
    assert forecast == pytest.approx(y[-6:], abs=1e-6)


# -- fitted state, parity, determinism -------------------------------------------------


@pytest.mark.component
def test_a_fitted_forecaster_saves_as_json_and_forecasts_new_data(tmp_path):
    hourly, _ = _hourly()
    spec = fs.Pipeline(
        [fs.forecast.Structural("flow", horizon="1D", seasonal=SEASONS)],
        name="flow_forecast",
    ).spec
    train = hourly.values.loc[:"2025-01-15"]
    fitted = fs.fit(spec, fs.SignalFrame.from_frame(train, units=hourly.units))
    json.dumps(fitted.to_payload())
    loaded = fs.FittedPipeline.load(fitted.save(tmp_path / "fitted"))
    new = loaded.transform(hourly)
    assert new.meta["flow"].issue_time == "2025-01-19T23:00:00+00:00"
    again = fitted.transform(hourly)
    assert new.values.equals(again.values)


@pytest.mark.component
def test_forecasts_are_identical_from_python_and_json_and_across_runs():
    hourly, _ = _hourly(days=10)
    steps = [
        fs.forecast.Regression("flow", horizon="12h", fourier=SEASONS, lags=["1h"]),
        fs.forecast.Structural("flow", horizon="12h", seasonal=SEASONS[:1], name="s"),
    ]
    for step in steps:
        spec = fs.Pipeline([step], name="p").spec
        from_json = fs.load_spec(json.loads(json.dumps(fs.dump_spec(spec))))
        first = fs.run(spec, hourly).signal
        second = fs.run(from_json, hourly).signal
        assert first.equals(second)


# -- what they refuse ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantiles": [0.125]}, "whole percent"),
        ({"quantiles": [1.0]}, "whole percent"),
        ({"exog": ["y"]}, "both forecast and used as exog"),
        ({"horizon": "-1h"}, "must be positive"),
    ],
)
def test_bad_options_are_refused(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        fs.forecast.Structural("y", **{"horizon": "1h", **kwargs})


@pytest.mark.unit
def test_method_specific_options_are_checked():
    with pytest.raises(ValidationError, match="uses no exog"):
        fs.forecast.Persistence("y", horizon="1h", exog=["x"])
    with pytest.raises(ValidationError, match="shorter than the horizon"):
        fs.forecast.Regression("y", horizon="6h", past=["x"], past_lags=["1h"])
    with pytest.raises(ValidationError, match="go together"):
        fs.forecast.Regression("y", horizon="6h", past=["x"])
    with pytest.raises(ValidationError, match="give fourier terms"):
        fs.forecast.Regression("y", horizon="6h")
    with pytest.raises(ValidationError, match="linear model"):
        fs.forecast.Regression("y", horizon="6h", lags=["1h"], model="scipy:curve_fit")


@pytest.mark.unit
def test_the_data_must_be_regular_and_the_times_on_its_grid():
    sf = _series(np.arange(48.0))
    with pytest.raises(SignalConfigError, match="whole number of the data's"):
        _forecast(fs.forecast.Persistence("y", horizon="90min"), sf)
    with pytest.raises(SignalConfigError, match="isn't on the data's"):
        _forecast(
            fs.forecast.Persistence("y", horizon="1h", issue_time="2025-01-01T10:30Z"),
            sf,
        )
    irregular = sf.values.drop(sf.index[5])
    with pytest.raises(SignalDataError, match="regular time index"):
        _forecast(fs.forecast.Persistence("y", horizon="1h"), fs.SignalFrame(irregular))
    with pytest.raises(SignalConfigError, match="unknown stream"):
        _forecast(fs.forecast.Persistence("z", horizon="1h"), sf)
    with pytest.raises(SignalDataError, match="no values to forecast from"):
        _forecast(fs.forecast.Persistence("y", horizon="1h"), _series([np.nan] * 5))
