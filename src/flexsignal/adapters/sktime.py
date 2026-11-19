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

import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    ModelName,
    model_class,
    model_option,
    regular_step,
    require,
    steps_in,
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

_QUANTILE_KEY = re.compile(r"q\d\d")


# -- the history as the library wants it ------------------------------------------


def _naive(index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DatetimeIndex:
    """The index as naive UTC times carrying the data's frequency."""
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(index, freq=pd.tseries.frequencies.to_offset(step))


class _Prepared:
    """The target's history and the exog streams, ready for a library."""

    def __init__(self, history: pd.DataFrame, future: pd.DataFrame, method: str):
        self.stream = str(history.columns[0])
        self.step = regular_step(history.index, method)
        target = history.iloc[:, 0].astype("float64")
        valid = np.flatnonzero(target.notna().to_numpy())
        if len(valid) < 2:
            raise SignalDataError(
                f"{method}: {self.stream!r} has fewer than two values up to the "
                "issue time, so there is nothing to fit on."
            )
        first, last = int(valid[0]), int(valid[-1])
        y = target.iloc[first : last + 1].interpolate(method="linear")
        self.y = pd.Series(y.to_numpy(), index=_naive(y.index, self.step))
        """The target from its first to its last value, gaps interpolated."""
        self.n_ahead = len(target) - 1 - last + len(future)
        """How many steps after the last value the forecast reaches."""
        self.future_index = _naive(future.index, self.step)
        self.exog = list(history.columns[1:])
        self.covariates: pd.DataFrame | None = None
        """The exog streams from the target's first value to the forecast's end."""
        if self.exog:
            both = pd.concat([history[self.exog].iloc[first:], future[self.exog]])
            both = both.astype("float64").interpolate(limit_direction="both")
            empty = [c for c in self.exog if both[c].isna().all()]
            if empty:
                raise SignalDataError(
                    f"{method}: exog stream {empty[0]!r} has no values to use."
                )
            self.covariates = pd.DataFrame(
                both.to_numpy(), index=_naive(both.index, self.step), columns=self.exog
            )

    def check_times(self, times: pd.Index, method: str) -> None:
        """Check a library's forecast times are the forecast rows."""
        got = pd.DatetimeIndex(times[-len(self.future_index) :]).as_unit("ns")
        if len(times) != self.n_ahead or not np.array_equal(
            got.asi8, self.future_index.as_unit("ns").asi8
        ):
            raise SignalDataError(  # pragma: no cover - a library bug
                f"{method}: the library forecast times {got[0]}..{got[-1]}, not the "
                f"forecast rows {self.future_index[0]}..{self.future_index[-1]}."
            )


class _Forecast(Mapping):
    """A forecast's mean, with quantiles computed only when they are read.

    flex-signal asks for ``"q10"``-style keys only when the step has
    ``quantiles``. When the model gives no intervals, asking for ``"std"`` (the
    last thing flex-signal tries) raises an error that says why.
    """

    def __init__(
        self,
        mean: np.ndarray,
        quantile: Callable[[float], np.ndarray] | None,
        refusal: str,
    ):
        self._mean = mean
        self._quantile = quantile
        self._refusal = refusal
        self._cache: dict[str, np.ndarray] = {}

    def __contains__(self, key: object) -> bool:
        if key == "mean":
            return True
        if isinstance(key, str) and _QUANTILE_KEY.fullmatch(key):
            return self._quantile is not None
        if key == "std" and self._quantile is None:
            raise SignalConfigError(self._refusal, field="quantiles")
        return False

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "mean":
            return self._mean
        if self._quantile is not None and _QUANTILE_KEY.fullmatch(str(key)):
            if key not in self._cache:
                self._cache[key] = self._quantile(int(key[1:]) / 100)
            return self._cache[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("mean",))

    def __len__(self) -> int:
        return 1


def _season_steps(season: str | None, step: pd.Timedelta, method: str) -> int | None:
    """How many data steps ``season`` covers, or None without a season."""
    if season is None:
        return None
    try:
        pd.Timedelta(season)
    except (TypeError, ValueError):
        raise SignalConfigError(
            f"{method}: season {season!r} isn't a duration, e.g. 1D, or null.",
            field="season",
        ) from None
    return steps_in(season, step, field="season", method=method)


@contextmanager
def _quiet() -> Iterator[None]:
    """Hide sktime's own deprecation warnings, which a user can't act on."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
        yield


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
    if season is None:
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
    if not isinstance(params, Mapping):
        raise SignalConfigError(
            f"{NAME}: params must be a dict of {model} arguments.", field="params"
        )
    names = cls.get_param_names()
    for key in params:
        if key in _RESERVED:
            raise SignalConfigError(
                f"{NAME}: params {key!r} takes a Python object, so a spec can't set "
                "it; remove it.",
                field="params",
                value=key,
            )
        if key not in names:
            raise SignalConfigError(
                f"{NAME}: params {key!r} isn't an argument of sktime's {model}; it "
                f"takes {', '.join(n for n in names if n not in _RESERVED)}.",
                field="params",
                value=key,
            )
    kwargs = _defaults(model, season, params)
    kwargs.update(params)
    for key in ("order", "seasonal_order"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    if "random_state" in names:
        kwargs.setdefault("random_state", seed)
    try:
        with _quiet():
            return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SignalConfigError(
            f"{NAME}: sktime's {model} refused its arguments {kwargs}: {exc}",
            field="params",
        ) from None


def _takes_exog(estimator: Any) -> bool:
    tag = estimator.get_tag("capability:exogenous", None, raise_error=False)
    if tag is not None:
        return bool(tag)
    return not estimator.get_tag("ignores-exogeneous-X", True, raise_error=False)


def _exog_models() -> str:
    return "SARIMAX or UnobservedComponents"


def _interval_models() -> str:
    return "NaiveForecaster, ThetaForecaster, AutoETS, SARIMAX or UnobservedComponents"


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
    prepared = _Prepared(history, future, NAME)
    m = _season_steps(season, prepared.step, NAME)
    base = require("sktime.forecasting.base", EXTRA, method=NAME)
    estimator = _build(model, params, m, seed)
    if prepared.exog and not _takes_exog(estimator):
        raise SignalConfigError(
            f"{NAME}: sktime's {model} takes no exog streams (X); remove exog or "
            f"use {_exog_models()}.",
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
        f"remove quantiles or use {_interval_models()}."
    )
    return _Forecast(mean, quantile, refusal)


register_forecaster(NAME, sktime_forecast)

for _model, _description in SKTIME_MODELS.items():
    register_backend(
        f"sktime:{_model}",
        f"{_MODULES[_model]}:{_model}",
        extra=EXTRA,
        description=_description,
        docs_url="https://www.sktime.net/en/stable/api_reference/forecasting.html",
    )
