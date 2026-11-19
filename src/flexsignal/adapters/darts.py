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
import re
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
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

_QUANTILE_KEY = re.compile(r"q\d\d")


# -- shared by the steps: the history as the library wants it -------------------------


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
        self.n_trail = len(target) - 1 - last
        """Missing values at the end of the history, forecast along with the rows."""
        self.n_ahead = self.n_trail + len(future)
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

    def check_times(self, times: pd.DatetimeIndex, method: str) -> None:
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
    """Hide darts' own deprecation warnings, which a user can't act on."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
        warnings.filterwarnings(
            "ignore", message=r".*extrapolate_trend='freq'", category=FutureWarning
        )
        yield


# -- the darts model --------------------------------------------------------------


def _check_params(
    params: Mapping[str, Any], cls: type, model: str, *, reserved: set[str]
) -> None:
    if not isinstance(params, Mapping):
        raise SignalConfigError(
            f"{NAME}: params must be a dict of {model} arguments.", field="params"
        )
    sig = inspect.signature(cls.__init__)
    names = [
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind is not p.VAR_KEYWORD
    ]
    names += _passed_on(model)
    for key in params:
        if key in reserved:
            raise SignalConfigError(
                f"{NAME}: params {key!r} is set by flex-signal; remove it.",
                field="params",
            )
        if key not in names:
            raise SignalConfigError(
                f"{NAME}: params {key!r} isn't an argument of darts' {model}; it takes "
                f"{', '.join(n for n in names if n not in reserved)}.",
                field="params",
                value=key,
            )


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
    _check_params(params, cls, model, reserved={"add_encoders"})
    kwargs: dict[str, Any] = {}
    if model in ("Theta", "FourTheta"):
        kwargs["season_mode"] = "additive"
    if model == "ARIMA":
        kwargs.update(p=1, d=0, q=0)
        if season is not None:
            kwargs["seasonal_order"] = (1, 1, 1, season)
    if season is not None and model in _SEASON_ARGS:
        kwargs[_SEASON_ARGS[model]] = season
    if model == "LinearRegressionModel":
        if has_exog:
            kwargs["lags_future_covariates"] = [0]
        if "lags" not in kwargs and "lags" not in params:
            raise SignalConfigError(
                f"{NAME}: LinearRegressionModel needs lags; set season (e.g. 1D) or "
                "params lags.",
                field="params",
            )
    kwargs.update(params)
    kwargs = _enum_params(kwargs, model)
    for key in ("seasonal_order",):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    if "random_state" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("random_state", seed)
    try:
        with _quiet():
            return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SignalConfigError(
            f"{NAME}: darts' {model} refused its arguments {kwargs}: {exc}",
            field="params",
        ) from None


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
    prepared = _Prepared(history, future, NAME)
    m = _season_steps(season, prepared.step, NAME)
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
    return _Forecast(mean, quantile, refusal)


register_forecaster(NAME, darts_forecast)

for _model, _description in DARTS_MODELS.items():
    register_backend(
        f"darts:{_model}",
        f"darts.models:{_model}",
        extra=EXTRA,
        description=_description,
        docs_url="https://unit8co.github.io/darts/generated_api/darts.models.html",
    )
