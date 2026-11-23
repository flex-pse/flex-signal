# Forecast and backtest

This guide forecasts a day of plant flow after cleaning it, scores the forecast against a seasonal-naive baseline
on the data's past, and lines the forecast up with a model's hourly steps. See
[Forecasting and backtesting](../explanation/forecasting_and_backtesting.md) for how it works.

## 1. A pipeline that cleans, then forecasts

The forecast step goes last. It needs a regular index, so resample the 5-minute data to hours first:

```python
import flexsignal as fs
from flexsignal.testing import synthetic

case = synthetic.plant(seed=1, periods=14 * 288).with_spikes("flow", count=10)
seasons = [{"period": "1D", "harmonics": 2}, {"period": "7D", "harmonics": 1}]

pipe = fs.Pipeline(name="flow_forecast")
pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6))
pipe.add_step(fs.select.Resample(freq="1h"))
pipe.add_step(
    fs.forecast.Structural("flow", horizon="1D", seasonal=seasons, quantiles=[0.1, 0.9])
)
```

## 2. Forecast

```python
result = fs.run(pipe.spec, case.signal())
forecast = result.signal
ahead = forecast.has_flag(fs.Flag.FORECAST)["flow"]
print(forecast.meta["flow"].issue_time)                       # the last hour with data
print(forecast.values.loc[ahead, ["flow__q10", "flow", "flow__q90"]].head())
```

The forecast rows are flagged `FORECAST`; `flow__q10` and `flow__q90` hold the 80% interval. To forecast from an
earlier time, set `issue_time="2025-01-15T23:00:00Z"` on the step.

To forecast again tomorrow without refitting, keep the fitted pipeline and give it the newer data:

```python
fitted = result.fitted
fitted.save("flow_forecast", overwrite=True)
newer = fs.FittedPipeline.load("flow_forecast").transform(case.signal())
```

## 3. Backtest against baselines

```python
scores = fs.backtest(
    pipe.spec,
    case.signal(),
    folds=4,
    compare=[
        fs.forecast.SeasonalNaive("flow", horizon="1D", season="1D"),
        fs.forecast.Regression("flow", horizon="1D", fourier=seasons),
    ],
)
print(scores.summary())   # rmse, mae, mape, coverage and skill per method
assert scores.summary().set_index("method").loc["forecast_structural", "skill"] > 0
```

The first `compare` step is the baseline for `skill` unless `baseline=` names another. `scores.scores` has one row
per method and fold; `scores.forecasts` has every forecast value with its actual value and lead time.

## 4. Into a flex-pse model's steps

A forecast lines up with a model's time steps like measured data. `to_grid` needs no flex-pse; the model's grid
here is a day of hourly steps after the data ends:

```python
import pandas as pd
from flexsignal import handoff

grid = pd.date_range("2025-01-20", periods=24, freq="1h")    # the model's day, naive local time
values = handoff.to_grid(forecast, grid, handoff.GridOptions(timezone="UTC"), columns=["flow"])
print(values.estimated)                                     # {'flow': 24}: every step is forecast
overrides = values.to_overrides(rename={"flow": "site.pump.flow_in"})
```

Then fix them with `fsp.apply_values(spec_or_model, overrides)`, as in
[Apply data to a flex-pse model](apply_to_a_flexpse_model.md).

## Variations

- **A value known ahead**, such as a price schedule: put its future rows in the input and add
  `exog=["price"]`. Every stream except `exog` is hidden after each issue time in a backtest.
- **Yesterday's flow as an input**: `fs.forecast.Regression("flow", horizon="1D", lags=["1D"], fourier=...)`
  forecasts one step at a time.
- **An upstream stream known only up to now**: `past=["upstream"], past_lags=["1D"]`, with every lag at least the
  horizon.
- **Train on the last week only**: `fs.backtest(..., train="7D")`.
