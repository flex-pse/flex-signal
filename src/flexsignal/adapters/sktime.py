"""sktime forecasters as a forecast step: ``fs.forecast.SktimeForecast``.

`sktime <https://www.sktime.net/>`_ gives many forecasting algorithms one
interface. This module registers one forecast step, ``SktimeForecast``, whose
``model`` option picks one of a short list of sktime forecasters that need only
sktime and statsmodels (no pmdarima, StatsForecast or Prophet). Install it with
``pip install flex-signal[sktime]``.

The forecaster is fitted on the history it is given each time the step runs,
and nothing is stored, so a fitted pipeline saves as plain JSON. Importing this
module registers the step and the ``sktime:<Forecaster>`` backend keys; sktime
itself is imported only when the step runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from functools import lru_cache
from types import SimpleNamespace
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

__all__ = ["SKTIME_MODELS", "sktime_forecast"]

NAME = "SktimeForecast"
EXTRA = "sktime"

_MODULES = {
    "NaiveForecaster": "sktime.forecasting.naive",
    "ThetaForecaster": "sktime.forecasting.theta",
    "ExponentialSmoothing": "sktime.forecasting.exp_smoothing",
    "AutoETS": "sktime.forecasting.ets",
    "SARIMAX": "sktime.forecasting.sarimax",
    "UnobservedComponents": "sktime.forecasting.structural",
    "STLForecaster": "sktime.forecasting.trend",
    "PolynomialTrendForecaster": "sktime.forecasting.trend",
}
"""Where each tested forecaster lives."""

SKTIME_MODELS: dict[str, str] = {
    "NaiveForecaster": (
        "Repeats the last value, or the value one season earlier. A baseline that "
        "gives intervals."
    ),
    "ThetaForecaster": "The Theta method on deseasonalised data. Gives intervals.",
    "ExponentialSmoothing": "Holt-Winters exponential smoothing (statsmodels).",
    "AutoETS": "Error-trend-season state-space smoothing (statsmodels). Gives "
    "intervals.",
    "SARIMAX": "Seasonal ARIMA with exog streams (statsmodels). Gives intervals.",
    "UnobservedComponents": (
        "A structural model: level, trend and seasonal cycles, with exog streams "
        "(statsmodels). Gives intervals."
    ),
    "STLForecaster": "Splits trend and season with STL and forecasts each.",
    "PolynomialTrendForecaster": "A polynomial trend in time, fitted by least "
    "squares.",
}
"""The sktime forecasters flex-signal is tested with, one line on each. Any
other sktime forecaster runs too, by its class name."""

OPTION = model_option(
    method=NAME,
    option="model",
    library="sktime",
    tested=tuple(SKTIME_MODELS),
    catalogue="https://www.sktime.net/en/stable/api_reference/forecasting.html",
    note="Forecasters built from other estimators (pipelines, reducers) can't be "
    "given in a spec.",
)

_RESERVED = {
    "callback",
    "forecaster_resid",
    "forecaster_seasonal",
    "forecaster_trend",
    "n_jobs",
    "regressor",
}
"""Arguments a spec can't set: they take Python objects or start processes."""

_SEASONAL = (
    "NaiveForecaster",
    "ThetaForecaster",
    "STLForecaster",
    "ExponentialSmoothing",
    "AutoETS",
    "SARIMAX",
    "UnobservedComponents",
)
"""The forecasters that use ``season``."""

_EXOG_MODELS = "SARIMAX or UnobservedComponents"
_INTERVAL_MODELS = (
    "NaiveForecaster, ThetaForecaster, AutoETS, SARIMAX or UnobservedComponents"
)


def _quiet() -> AbstractContextManager[None]:
    """Hide sktime's own deprecation warnings, which a user can't act on."""
    return quiet_pandas()


# -- the sktime forecaster ------------------------------------------------------------


def _defaults(model: str, season: int | None, params: Mapping[str, Any]) -> dict:
    """The arguments flex-signal sets before ``params``, from ``season``."""
    kwargs: dict[str, Any] = {}
    if model == "ThetaForecaster":
        kwargs["deseasonalize_model"] = "additive"
    if model == "SARIMAX":
        kwargs["order"] = (1, 0, 0)
    if model == "UnobservedComponents":
        kwargs["level"] = "local level"
    if season is None or season < 2:  # a one-step season is no cycle
        return kwargs
    if model in ("NaiveForecaster", "ThetaForecaster", "STLForecaster"):
        kwargs["sp"] = season
    elif model == "ExponentialSmoothing":
        kwargs.update(sp=season, seasonal="add")
    elif model == "AutoETS":
        kwargs["sp"] = season
        if not params.get("auto"):
            kwargs["seasonal"] = "add"
    elif model == "SARIMAX":
        kwargs["seasonal_order"] = (1, 1, 1, season)
    elif model == "UnobservedComponents":
        kwargs["freq_seasonal"] = [{"period": season, "harmonics": min(3, season // 2)}]
    return kwargs


@lru_cache(maxsize=1)
def _all_forecasters() -> SimpleNamespace:
    """Sktime's own registry of forecasters, by class name (slow: done once)."""
    registry = require("sktime.registry", EXTRA, method=NAME)
    return SimpleNamespace(
        **dict(registry.all_estimators(estimator_types="forecaster"))
    )


def _build(model: str, params: Mapping[str, Any], season: int | None, seed: int) -> Any:
    base = require("sktime.forecasting.base", EXTRA, method=NAME).BaseForecaster
    if model in _MODULES:
        cls = getattr(require(_MODULES[model], EXTRA, method=NAME), model)
    else:
        cls = model_class(_all_forecasters(), model, OPTION, base=base)
    settings = _defaults(model, season, params)
    if "random_state" in cls.get_param_names():
        settings["random_state"] = seed
    params = dict(params)
    for key in ("order", "seasonal_order"):
        if isinstance(params.get(key), list):
            params[key] = tuple(params[key])
    with _quiet():
        return build_model(
            cls,
            settings,
            params,
            OPTION,
            managed=_RESERVED,
            why_managed="those take Python objects or start processes, which a "
            "spec can't give; remove them",
        )


def _takes_exog(estimator: Any) -> bool:
    tag = estimator.get_tag("capability:exogenous", None, raise_error=False)
    if tag is not None:
        return bool(tag)
    return not estimator.get_tag("ignores-exogeneous-X", True, raise_error=False)


def sktime_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: ModelName = "ThetaForecaster",
    params: dict[str, Any] | None = None,
    season: str | None = "1D",
    seed: int = 0,
) -> Mapping[str, np.ndarray]:
    """Forecast with an sktime forecaster, fitted on the history each time it runs.

    Use it to try sktime's statistical forecasters (Theta, exponential
    smoothing, ETS, seasonal ARIMA, a structural model, STL) on a signal,
    scored in ``fs.backtest`` against flex-signal's own forecasters. Gaps in the
    history are interpolated for fitting; put an impute step first to fill them
    better. ``SARIMAX`` and ``UnobservedComponents`` take ``exog`` streams as
    sktime's ``X``; the others refuse them. Quantiles come from sktime's
    ``predict_quantiles`` for the forecasters that give intervals
    (NaiveForecaster, ThetaForecaster, AutoETS, SARIMAX, UnobservedComponents);
    the others refuse them.

    Args:
        history: The target (first column) and exog streams up to the issue time.
        future: The exog streams over the forecast rows.
        model: The sktime forecaster, by class name. Tested: NaiveForecaster,
            ThetaForecaster, ExponentialSmoothing, AutoETS, SARIMAX,
            UnobservedComponents, STLForecaster and PolynomialTrendForecaster.
            Any other sktime forecaster, e.g. TrendForecaster or BATS, runs
            too, untested: its own arguments go in params.
        params: Arguments for the forecaster's constructor, e.g. {"strategy":
            "mean"} for NaiveForecaster or {"order": [2, 0, 1]} for SARIMAX. They
            override the ones flex-signal derives from season.
        season: The signal's cycle, e.g. 1D for a daily cycle, or null for none.
            It sets sp (NaiveForecaster, ThetaForecaster, STLForecaster), sp and
            seasonal="add" (ExponentialSmoothing, AutoETS), seasonal_order
            (1, 1, 1, m) (SARIMAX, whose order defaults to (1, 0, 0)) or a
            freq_seasonal cycle (UnobservedComponents, with a local level).
            PolynomialTrendForecaster doesn't use it.
        seed: The random_state of forecasters that take one, so a run repeats.

    Returns:
        The forecast's mean, and its quantiles when the forecaster gives
        intervals.
    """
    params = dict(params or {})
    prepared = Prepared(history, future, NAME)
    m = season_steps(season, prepared.step, NAME, used=model in _SEASONAL)
    base = require("sktime.forecasting.base", EXTRA, method=NAME)
    estimator = _build(model, params, m, seed)
    if prepared.exog and not _takes_exog(estimator):
        raise SignalConfigError(
            f"{NAME}: sktime's {model} takes no exog streams (X); remove exog or "
            f"use {_EXOG_MODELS}.",
            field="exog",
        )
    n_fit = len(prepared.y)
    X_fit = X_ahead = None
    if prepared.covariates is not None:
        X_fit = prepared.covariates.iloc[:n_fit]
        X_ahead = prepared.covariates.iloc[n_fit:]
    fh = base.ForecastingHorizon(np.arange(1, prepared.n_ahead + 1), is_relative=True)
    with _quiet():
        try:
            estimator.fit(prepared.y, X=X_fit, fh=fh)
            prediction = estimator.predict(X=X_ahead)
        except ValueError as exc:
            raise SignalDataError(
                f"{NAME}: sktime's {model} could not forecast {prepared.stream!r}: "
                f"{exc}"
            ) from exc
    prepared.check_times(prediction.index, NAME)
    n = len(future)
    mean = np.asarray(prediction, dtype="float64").reshape(-1)[-n:]

    quantile = None
    if estimator.get_tag("capability:pred_int", False, raise_error=False):

        def quantile(q: float) -> np.ndarray:
            with _quiet():
                table = estimator.predict_quantiles(X=X_ahead, alpha=[q])
            return table.to_numpy(dtype="float64")[-n:, 0]

    refusal = (
        f"{NAME}: sktime's {model} gives no intervals, so it can't write quantiles; "
        f"remove quantiles or use {_INTERVAL_MODELS}."
    )
    return Forecast(mean, quantile, refusal)


register_forecaster(NAME, sktime_forecast)

for _model, _description in SKTIME_MODELS.items():
    register_backend(
        f"sktime:{_model}",
        f"{_MODULES[_model]}:{_model}",
        extra=EXTRA,
        description=_description,
        docs_url="https://www.sktime.net/en/stable/api_reference/forecasting.html",
    )
