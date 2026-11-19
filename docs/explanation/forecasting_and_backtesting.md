# Forecasting and backtesting

A forecast is a prediction of a stream's future values. In flex-signal it is a pipeline step like any other: it
runs after the cleaning steps, from Python or a JSON spec, and its fitted state saves as JSON. This page explains
where a forecast goes in the data, what it can use, and how a backtest scores it fairly.

## A forecast fills the future rows

A forecast step is made at an **issue time**. It learns from the target's values up to that time and writes its
forecast into the rows after it, up to `issue_time + horizon`. Where the data stops sooner, the step extends the
time index on the data's own step. The new rows are:

- flagged `FORECAST` in the target stream, so every later step and reader can tell them from measurements;
- in the target's units, with the issue time recorded in the target's metadata (`sf.meta["flow"].issue_time`);
- joined by one stream per quantile, e.g. `flow__q10` and `flow__q90`, when `quantiles` is set.

By default the issue time is the last time any target has a value, which is "now" for a live run: a fitted
pipeline's `transform` forecasts from the newest data it is given. Set `issue_time` to forecast from an earlier
time. Target values after the issue time are never used, and in the forecast rows they are replaced.

The data needs a regular time index, and `horizon` and `issue_time` must fall on it. Put an `align` or
`resample` step before the forecast, as the forecast usually wants a coarser step than the raw data anyway.

## What a forecast can use

| Input | Known | Example | Field |
|---|---|---|---|
| The target's own history | up to the issue time | yesterday's flow | every method |
| Cycles | always | time of day, day of week | `seasonal` (`structural`), `fourier` (`regression`) |
| Streams known ahead | through the horizon | a price schedule, a weather forecast | `exog` |
| Other streams known only up to the issue time | up to the issue time | upstream flow | `past` with `past_lags` (`regression`) |

**Streams known ahead** are simply rows after the issue time in the input data. If one of them is missing within
the horizon, the forecast stops with an error that names the stream and the time, rather than guessing.

**Streams known only up to the issue time** can be used only at a lag at least as long as the horizon, so their
value is already known at every forecast step. `regression` checks this when the step is built.

## The methods

| Method | What it is | Intervals |
|---|---|---|
| `forecast.Persistence` | The last value, held | Widen with the square root of the lead time |
| `forecast.SeasonalNaive` | The value one season earlier (`season="1D"`: the same time yesterday) | Widen with the square root of the seasons reached back |
| `forecast.Structural` | statsmodels' `UnobservedComponents`: a level or trend, seasonal cycles, an optional AR term, `exog` | From the model's forecast variance |
| `forecast.Regression` | A linear model of cycles, `exog`, the target's lags (forecast step by step) and `past` streams | The training residuals' quantiles; they don't widen with lead time |

`Persistence` and `SeasonalNaive` are **baselines**. They have no parameters to speak of, and a model that can't
beat them isn't worth its complexity.

`Structural` estimates its parameters by maximum likelihood when it is fitted, and stores them as JSON. When it
forecasts, it runs the Kalman filter over the new data's history with those parameters, so it uses the newest
data without being refitted. Missing target values in the history are fine. Whether the fit converged is in the
fit report.

Forecasting libraries that require pandas 2 (skforecast, statsforecast, mlforecast) aren't used, because
flex-signal and flex-pse require pandas 3. Their adapters wait until they support it. Until then, a forecaster of
your own can be registered (see the extension API in the plan).

## Backtesting without peeking

A forecast is only as good as its errors on data it hasn't seen. {func}`flexsignal.eval.backtest` measures them on the
data's own past:

1. It runs the steps before the forecast step once on all the data, to find the forecast's grid and the values to
   score against.
2. It picks `folds` issue times, blocked and in time order. The last fold's forecast ends at the target's last
   value, and each earlier one is issued `every` before the next (by default, one horizon, so the forecasts don't
   overlap).
3. For each fold, it hides every stream except `exog` after the issue time, flagging the hidden rows `MISSING`
   and `NO_SAMPLE` so no cleaning or imputation step fills them in. Then it fits and runs the **whole** pipeline,
   cleaning included, with the forecast issued at that time.
4. It scores each forecast (RMSE, MAE, MAPE, and the share of values inside the outermost quantiles).

Hiding the data rather than trusting each step to ignore it is what makes the score honest. A centred cleaning
window, or an imputer that interpolates across the issue time, would otherwise pull information back from the
future.

`compare=[...]` swaps other forecast steps into the same pipeline, so a model and its baselines are scored on
exactly the same cleaned data. {meth}`BacktestResult.summary <flexsignal.eval.BacktestResult.summary>` averages
the folds and adds **skill**: `1 - rmse / baseline rmse`. Above 0 beats the baseline; 1 is perfect.

## Into a model

A forecast is a prediction, so it reaches a flex-pse model as fixed values, exactly as measured data does:
{func}`flexsignal.handoff.to_grid` lines it up with the model's time steps, and
{func}`flexsignal.integrations.flexpse.apply_values` fixes the model's variables to it. `GridValues.estimated`
counts the steps that rest on forecast values. See [Handing data to flex-pse models](flexpse_integration.md).
