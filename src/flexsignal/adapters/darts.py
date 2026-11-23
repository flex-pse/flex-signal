"""darts forecasting models as a forecast step: ``fs.forecast.DartsForecast``.

`darts <https://unit8co.github.io/darts/>`_ wraps many forecasting models behind
one interface. This module registers one forecast step, ``DartsForecast``, whose
``model`` option picks one of a short list of darts models that run on a CPU in
seconds and need nothing beyond darts and statsmodels (no torch, no Prophet, no
StatsForecast). Install it with ``pip install flex-signal[darts]``.

The model is fitted on the history it is given each time the step runs, and
nothing is stored, so a fitted pipeline saves as plain JSON. Importing this
module registers the step and the ``darts:<Model>`` backend keys; darts itself
is imported only when the step runs.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    Forecast,
    ModelName,
    Prepared,
    build_model,
    model_class,
    model_option,
    quiet_pandas,
    require,
    season_steps,
)
from flexsignal.backends import register_backend
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.steps.registered import register_forecaster

__all__ = ["DARTS_MODELS", "darts_forecast"]

NAME = "DartsForecast"
EXTRA = "darts"

DARTS_MODELS: dict[str, str] = {
    "ExponentialSmoothing": (
        "Holt-Winters exponential smoothing (statsmodels): level, trend and a "
        "seasonal cycle. Gives intervals."
    ),
    "Theta": "The Theta method: a damped trend on deseasonalised data.",
    "FourTheta": "Theta with a choice of trend and seasonality modes.",
    "ARIMA": "Seasonal ARIMA (statsmodels). Takes exog streams; gives intervals.",
    "KalmanForecaster": (
        "A linear state-space model fitted with N4SID. Takes exog streams; gives "
        "intervals."
    ),
    "LinearRegressionModel": (
        "Linear regression on the target's lagged values and the exog streams. "
        "Gives intervals with likelihood='quantile'."
    ),
    "NaiveSeasonal": "Repeats the value one season earlier. A baseline.",
    "NaiveDrift": "A straight line from the first to the last value. A baseline.",
    "NaiveMean": "The mean of the history. A baseline.",
    "FFT": "Repeats the strongest frequencies of the history's Fourier transform.",
}
"""The darts models flex-signal is tested with, one line on each. Any other
darts forecasting model runs too, by its class name."""

OPTION = model_option(
    method=NAME,
    option="model",
    library="darts",
    tested=tuple(DARTS_MODELS),
    catalogue="https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html",
    note="Models built from other models (ensembles) can't be given in a spec.",
)

_SEASON_ARGS = {
    "ExponentialSmoothing": "seasonal_periods",
    "Theta": "seasonality_period",
    "FourTheta": "seasonality_period",
    "LinearRegressionModel": "lags",
    "NaiveSeasonal": "K",
}
"""The constructor argument ``season`` sets, per model (ARIMA: seasonal_order)."""

_ENUMS = {
    "ExponentialSmoothing": {"trend": "ModelMode", "seasonal": "SeasonalityMode"},
    "Theta": {"season_mode": "SeasonalityMode"},
    "FourTheta": {
        "season_mode": "SeasonalityMode",
        "model_mode": "ModelMode",
        "trend_mode": "TrendMode",
    },
}
"""Constructor arguments darts takes as enums; a spec gives their string values."""

_EXOG_MODELS = ("ARIMA", "KalmanForecaster", "LinearRegressionModel")
_PROBABILISTIC = (
    "ExponentialSmoothing",
    "ARIMA",
    "KalmanForecaster",
    'LinearRegressionModel with params {"likelihood": "quantile"}',
)

_SEASONAL = (*_SEASON_ARGS, "ARIMA")
"""The models that use ``season``."""


def _quiet() -> AbstractContextManager[None]:
    """Hide darts' own deprecation warnings, which a user can't act on."""
    return quiet_pandas((r".*extrapolate_trend='freq'", FutureWarning))


# -- the darts model --------------------------------------------------------------


def _passed_on(model: str) -> list[str]:
    """Arguments a darts model passes on to the library it wraps."""
    if model == "ExponentialSmoothing":
        holtwinters = require("statsmodels.tsa.holtwinters", EXTRA, method=NAME)
        target = holtwinters.ExponentialSmoothing.fit
    elif model == "LinearRegressionModel":
        linear = require("sklearn.linear_model", EXTRA, method=NAME)
        target = linear.LinearRegression.__init__
    else:
        return []
    return [n for n in inspect.signature(target).parameters if n != "self"]


def _enum_params(params: dict[str, Any], model: str) -> dict[str, Any]:
    utils = require("darts.utils.utils", EXTRA, method=NAME)
    out = dict(params)
    for key, enum_name in _ENUMS.get(model, {}).items():
        if key not in out:
            continue
        enum = getattr(utils, enum_name)
        if isinstance(out[key], enum):
            continue
        try:
            out[key] = enum(out[key])
        except ValueError:
            choices = [e.value for e in enum]
            raise SignalConfigError(
                f"{NAME}: params {key!r} of {model} is one of {choices}, not "
                f"{out[key]!r}.",
                field="params",
                value=out[key],
            ) from None
    return out


def _build(
    model: str,
    params: Mapping[str, Any],
    season: int | None,
    has_exog: bool,
    seed: int,
) -> Any:
    models = require("darts.models", EXTRA, method=NAME)
    base = require(
        "darts.models.forecasting.forecasting_model", EXTRA, method=NAME
    ).ForecastingModel
    cls = model_class(models, model, OPTION, base=base)
    # A one-step season is no cycle at all (statsmodels refuses it), but it is
    # still one lag for LinearRegressionModel.
    cycle = season if season is not None and season > 1 else None
    kwargs: dict[str, Any] = {}
    if model in ("Theta", "FourTheta"):
        kwargs["season_mode"] = "additive"
    if model == "ARIMA":
        kwargs.update(p=1, d=0, q=0)
        if cycle is not None:
            kwargs["seasonal_order"] = (1, 1, 1, cycle)
    if model == "LinearRegressionModel":
        if season is not None:
            kwargs["lags"] = season
        if has_exog:
            kwargs["lags_future_covariates"] = [0]
        if "lags" not in kwargs and "lags" not in params:
            raise SignalConfigError(
                f"{NAME}: LinearRegressionModel needs lags; set season (e.g. 1D) or "
                "params lags.",
                field="params",
            )
    elif cycle is not None and model in _SEASON_ARGS:
        kwargs[_SEASON_ARGS[model]] = cycle
    if "random_state" in inspect.signature(cls.__init__).parameters:
        kwargs["random_state"] = seed
    params = _enum_params(dict(params), model)
    if isinstance(params.get("seasonal_order"), list):
        params["seasonal_order"] = tuple(params["seasonal_order"])
    with _quiet():
        return build_model(
            cls,
            _enum_params(kwargs, model),
            params,
            OPTION,
            managed={"add_encoders"},
            extra_accepted=_passed_on(model),
            strict=True,
        )


def darts_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: ModelName = "ExponentialSmoothing",
    params: dict[str, Any] | None = None,
    season: str | None = "1D",
    num_samples: int = 500,
    seed: int = 0,
) -> Mapping[str, np.ndarray]:
    """Forecast with a darts model, fitted on the history each time it runs.

    Use it to try darts' classical models (exponential smoothing, Theta, ARIMA,
    a Kalman filter, a lagged linear regression) on a signal, scored in
    ``fs.backtest`` against flex-signal's own forecasters. Gaps in the history
    are interpolated for fitting; put an impute step first to fill them better.
    ``ARIMA``, ``KalmanForecaster`` and ``LinearRegressionModel`` take ``exog``
    streams as darts' future covariates; the others refuse them. Quantiles come
    from sampled forecasts of ``ExponentialSmoothing``, ``ARIMA`` and
    ``KalmanForecaster``, or ``LinearRegressionModel`` with
    ``params={"likelihood": "quantile"}``; other models refuse them.

    Args:
        history: The target (first column) and exog streams up to the issue time.
        future: The exog streams over the forecast rows.
        model: The darts model, by class name. Tested: ExponentialSmoothing,
            Theta, FourTheta, ARIMA, KalmanForecaster, LinearRegressionModel,
            NaiveSeasonal, NaiveDrift, NaiveMean and FFT. Any other darts
            forecasting model, e.g. AutoARIMA or NHiTSModel, runs too,
            untested: its own arguments go in params.
        params: Arguments for the darts model's constructor, e.g. {"trend": null}
            for ExponentialSmoothing or {"p": 2, "q": 1} for ARIMA. They override
            the ones flex-signal derives from season.
        season: The signal's cycle, e.g. 1D for a daily cycle, or null for none.
            It sets seasonal_periods (ExponentialSmoothing), seasonality_period
            (Theta, FourTheta), seasonal_order (1, 1, 1, m) (ARIMA, which also
            defaults to p=1, d=0, q=0), lags (LinearRegressionModel) or K
            (NaiveSeasonal). Other models don't use it.
        num_samples: How many sampled forecasts the quantiles are taken from,
            between 10 and 10000.
        seed: Seeds the sampling, so a run repeats exactly.

    Returns:
        The forecast's mean, and its quantiles when the model gives intervals.
    """
    params = dict(params or {})
    if not 10 <= num_samples <= 10_000:
        raise SignalConfigError(
            f"{NAME}: num_samples is between 10 and 10000, not {num_samples}.",
            field="num_samples",
        )
    prepared = Prepared(history, future, NAME)
    m = season_steps(season, prepared.step, NAME, used=model in _SEASONAL)
    darts = require("darts", EXTRA, method=NAME)
    estimator = _build(model, params, m, bool(prepared.exog), seed)
    if prepared.exog and not estimator.supports_future_covariates:
        raise SignalConfigError(
            f"{NAME}: darts' {model} takes no exog streams (future covariates); "
            f"remove exog or use {', '.join(_EXOG_MODELS)}.",
            field="exog",
        )
    with _quiet():
        series = darts.TimeSeries.from_times_and_values(
            prepared.y.index, prepared.y.to_numpy()
        )
        covariates = (
            None
            if prepared.covariates is None
            else darts.TimeSeries.from_dataframe(prepared.covariates)
        )
        extra = {} if covariates is None else {"future_covariates": covariates}
        try:
            estimator.fit(series, **extra)
            prediction = estimator.predict(prepared.n_ahead, **extra)
        except ValueError as exc:
            raise SignalDataError(
                f"{NAME}: darts' {model} could not forecast {prepared.stream!r}: "
                f"{exc}"
            ) from exc
    prepared.check_times(prediction.time_index, NAME)
    n = len(future)
    mean = prediction.values(copy=False)[-n:, 0].astype("float64")

    quantile = None
    if estimator.supports_probabilistic_prediction:
        samples: list[np.ndarray] = []

        def quantile(q: float) -> np.ndarray:
            if not samples:
                with _quiet():
                    sampled = estimator.predict(
                        prepared.n_ahead, num_samples=num_samples, **extra
                    )
                samples.append(sampled.all_values(copy=False)[-n:, 0, :])
            return np.quantile(samples[0], q, axis=1)

    probabilistic = ", ".join(_PROBABILISTIC)
    refusal = (
        f"{NAME}: darts' {model} gives no intervals, so it can't write quantiles; "
        f"remove quantiles or use {probabilistic}."
    )
    return Forecast(mean, quantile, refusal)


register_forecaster(NAME, darts_forecast)

for _model, _description in DARTS_MODELS.items():
    register_backend(
        f"darts:{_model}",
        f"darts.models:{_model}",
        extra=EXTRA,
        description=_description,
        docs_url="https://unit8co.github.io/darts/generated_api/darts.models.html",
    )
