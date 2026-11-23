"""Forecast steps: predict a stream's future values and write them into the frame.

A forecast step is made at an **issue time**. It learns from the target's values
up to that time and writes its forecast into the rows after it, up to
``issue_time + horizon``, extending the time index on the data's own step where
the data stops sooner. Those rows are flagged ``FORECAST``, and the target's
metadata records the issue time. Target values after the issue time are not
used: in the forecast rows they are replaced.

- ``persistence``: the last value, held. A baseline.
- ``seasonal_naive``: the value one season earlier, e.g. the same time
  yesterday. A baseline.
- ``structural``: a structural time-series model (statsmodels'
  ``UnobservedComponents``): a level or trend, seasonal cycles, an optional
  autoregressive term, and ``exog`` streams.
- ``regression``: a linear model of daily or weekly cycles, ``exog`` streams,
  the target's own recent values, and other streams' older values.

Every forecast step takes the same core fields:

- ``streams`` and ``horizon``: what to forecast, and how far ahead.
- ``issue_time``: when the forecast is made. By default, the last time any
  target has a value.
- ``exog``: streams known ahead, such as a price schedule or a weather forecast.
  Their values after the issue time are rows of the input data.
- ``quantiles``: e.g. ``[0.1, 0.9]``, written as the streams ``flow__q10`` and
  ``flow__q90`` with the same units as ``flow``.

The data needs a regular time index; run ``align`` or ``resample`` first.
:func:`flexsignal.eval.backtest` scores a forecast step on the data's past.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from flexsignal.backends import get_backend
from flexsignal.backends.models import (
    check_model_key,
    fit_linear,
    is_linear_model,
    predict_linear,
)
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.spec.models import Duration, TimestampStr, to_timedelta, to_utc
from flexsignal.steps._base import (
    StreamStep,
    check_streams,
    fourier_terms,
    one_is_a_list,
    registered_lookup,
    regular_step,
    structural_args,
    whole_steps,
)
from flexsignal.steps.impute import CycleTerm

__all__ = [
    "ForecastStep",
    "Persistence",
    "Regression",
    "SeasonalNaive",
    "Structural",
    "quantile_column",
]

_NOT_A_SAMPLE = FLAG_DTYPE(int(Flag.MISSING | Flag.NO_SAMPLE))


def quantile_column(stream: str, q: float) -> str:
    """Return the name of a quantile stream, e.g. ``flow__q10`` for 0.1."""
    return f"{stream}__q{round(q * 100):02d}"


def _normal_ppf(q: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(q))


@dataclass(frozen=True)
class _Window:
    """Where a forecast is made: the history, the forecast rows, and the inputs."""

    frame: SignalFrame
    """The input, with its index extended to cover the horizon (when forecasting;
    a window made for ``fit`` is the input as it is)."""
    issue: pd.Timestamp
    """The issue time."""
    step: pd.Timedelta
    """The data's step."""
    n_history: int
    """Rows at or before the issue time; the history is ``[:n_history]``."""
    future: np.ndarray
    """Positions of the forecast rows, issue time + 1 step to + horizon; empty in a
    window made for ``fit``."""

    def history(self, stream: str) -> np.ndarray:
        """The stream's values up to the issue time."""
        return self.frame.values[stream].to_numpy()[: self.n_history]

    def column(self, stream: str) -> np.ndarray:
        """The stream's values over the whole extended index."""
        return self.frame.values[stream].to_numpy()


class ForecastStep(StreamStep):
    """Base class of the forecast steps; the fields every forecaster shares.

    A subclass implements ``_fit_one(window, stream)``, returning a stream's
    JSON-able state, and ``_predict(window, stream, state)``, returning the
    forecast's mean and either its standard deviation or ``{q: values}`` for each
    requested quantile (``None`` only when no quantiles are requested).
    """

    horizon: Duration = Field(
        description="How far ahead to forecast, e.g. 1D. A whole number of steps."
    )
    issue_time: TimestampStr | None = Field(
        None,
        description=(
            "When the forecast is made: it uses the targets' values up to this time "
            "and forecasts the rows after it. Default: the last time any target has "
            "a value."
        ),
    )
    exog: tuple[str, ...] = Field(
        (),
        description=(
            "Streams known ahead (a price schedule, a weather forecast). Their "
            "values after the issue time must be in the data."
        ),
    )
    quantiles: tuple[float, ...] = Field(
        (),
        description=(
            "Quantiles to write as <stream>__q10-style streams, e.g. [0.1, 0.9]. "
            "Each is a whole percent between 1 and 99."
        ),
    )

    _one_exog_is_a_list = field_validator("exog", mode="before")(one_is_a_list)

    @field_validator("quantiles")
    @classmethod
    def _whole_percents(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        for q in value:
            if not 0.01 <= q <= 0.99 or abs(q * 100 - round(q * 100)) > 1e-9:
                raise ValueError(
                    f"quantile {q} must be a whole percent between 0.01 and 0.99, "
                    "e.g. 0.1 or 0.9"
                )
        if len(set(value)) != len(value):
            raise ValueError("quantiles must be unique")
        return tuple(sorted(value))

    @model_validator(mode="after")
    def _exog_isnt_a_target(self) -> ForecastStep:
        overlap = sorted(set(self.exog) & set(self.streams))
        if overlap:
            raise ValueError(f"{overlap} can't be both forecast and used as exog")
        return self

    # -- the window -------------------------------------------------------------

    def _window(self, sf: SignalFrame, *, ahead: bool = True) -> _Window:
        """Return where the forecast is made.

        With ``ahead`` (``transform``), the index is extended to cover the
        horizon, and the exog streams must have values over it. Without
        (``fit``), the window is only the history: a model is fitted on the past,
        so the training data needn't reach past the targets' last value.
        """
        check_streams(sf, self.streams)
        check_streams(sf, self.exog, field="exog")
        index = sf.index
        step = regular_step(index, kind=self.kind)
        horizon = to_timedelta(self.horizon)
        n_ahead = whole_steps(
            horizon, step, what=f"horizon {self.horizon}", field="horizon"
        )
        issue = self._issue_time(sf)
        off_grid = (
            f"issue_time {issue} isn't on the data's {step} grid starting {index[0]}."
        )
        if issue < index[0]:
            raise SignalConfigError(off_grid, field="issue_time")
        offset = whole_steps(
            issue - index[0],
            step,
            what="issue_time",
            field="issue_time",
            error=off_grid,
        )
        n_history = offset + 1
        if not ahead:
            return _Window(sf, issue, step, n_history, np.arange(0))
        end = issue + horizon
        if end > index[-1]:
            extra = pd.date_range(index[-1] + step, end, freq=step)
            values = sf.values.reindex(index.append(extra))
            flags = sf.flags.reindex(values.index, fill_value=_NOT_A_SAMPLE)
            sf = sf.evolve(values=values, flags=flags.astype(FLAG_DTYPE))
        future = np.arange(n_history, n_history + n_ahead)
        window = _Window(sf, issue, step, n_history, future)
        for col in self.exog:
            known = window.column(col)[future]
            if np.isnan(known).any():
                when = sf.index[future[np.isnan(known)][0]]
                raise SignalDataError(
                    f"{self.kind}: exog stream {col!r} has no value at {when}, within "
                    f"the horizon. Values known ahead must be in the data up to "
                    f"{end}."
                )
        return window

    def _issue_time(self, sf: SignalFrame) -> pd.Timestamp:
        if self.issue_time is not None:
            return to_utc(self.issue_time)
        valid = sf.values[list(self.streams)].notna().any(axis=1).to_numpy()
        if not valid.any():
            raise SignalDataError(
                f"{self.kind}: the targets {list(self.streams)} have no values to "
                "forecast from."
            )
        return sf.index[np.flatnonzero(valid)[-1]]

    def _exog_matrix(self, window: _Window) -> np.ndarray:
        """The exog streams over the extended index, one column each."""
        if not self.exog:
            return np.empty((len(window.frame), 0))
        return np.column_stack([window.column(c) for c in self.exog])

    # -- fit and transform ----------------------------------------------------------

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Learn each target's model from the training data up to the issue time."""
        window = self._window(sf, ahead=False)
        states = {col: self._fit_one(window, col) for col in self.streams}
        out: dict[str, Any] = {"streams": {c: s for c, s in states.items()}}
        metrics = {c: s.pop("metrics") for c, s in states.items() if "metrics" in s}
        if metrics:
            out["metrics"] = metrics
        return out

    def _check_quantile_names(self, sf: SignalFrame) -> None:
        """Raise if a quantile stream would overwrite one of the input's streams.

        A stream that holds only forecast values (every value it has is flagged
        ``FORECAST``) is an earlier forecast's output, e.g. when a fitted
        pipeline runs again on its own output, and is replaced.
        """
        for col in self.streams:
            for q in self.quantiles:
                name = quantile_column(col, q)
                if name not in sf:
                    continue
                present = sf.values[name].notna().to_numpy()
                forecast = sf.has_flag(Flag.FORECAST)[name].to_numpy()
                if (forecast | ~present).all():
                    continue
                raise SignalConfigError(
                    f"the data already has a stream {name!r}, and the {q} quantile "
                    f"of the {col!r} forecast would overwrite it. Rename that stream "
                    "(a rename step before this one) or drop that quantile.",
                    field="quantiles",
                    value=name,
                )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Write the forecast into the rows after the issue time."""
        self._check_quantile_names(sf)
        window = self._window(sf)
        frame = window.frame
        values = frame.values
        flags = frame.flags
        meta = dict(frame.meta)
        issued = window.issue.isoformat()
        for col in self.streams:
            mean, sd = self._predict(window, col, state["streams"][col])
            column = values[col].to_numpy(copy=True)
            column[window.future] = mean
            values[col] = column
            col_flags = flags[col].to_numpy(copy=True)
            col_flags[window.future] = FLAG_DTYPE(int(Flag.FORECAST))
            flags[col] = col_flags
            meta[col] = meta[col].model_copy(update={"issue_time": issued})
            for q in self.quantiles:
                name = quantile_column(col, q)
                bound = np.full(len(frame), np.nan)
                if isinstance(sd, Mapping):
                    bound[window.future] = sd[q]
                else:
                    bound[window.future] = mean + _normal_ppf(q) * sd
                q_flags = np.full(len(frame), _NOT_A_SAMPLE, dtype=FLAG_DTYPE)
                q_flags[window.future] = FLAG_DTYPE(int(Flag.FORECAST))
                values[name] = bound
                flags[name] = q_flags
                meta[name] = meta[col].model_copy(
                    update={
                        "description": f"{round(q * 100)}% quantile of the {col} "
                        "forecast."
                    }
                )
        return frame.evolve(values=values, flags=flags, meta=meta)

    # -- for subclasses -------------------------------------------------------------

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def _predict(
        self, window: _Window, stream: str, state: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray | Mapping[float, np.ndarray] | None]:
        raise NotImplementedError  # pragma: no cover


def _no_exog(step: ForecastStep) -> None:
    if step.exog:
        raise ValueError(f"{step.kind} uses no exog streams; remove exog")


def _last_valid(window: _Window, stream: str) -> int:
    history = window.history(stream)
    valid = np.flatnonzero(~np.isnan(history))
    if not len(valid):
        raise SignalDataError(
            f"{stream!r} has no values up to the issue time {window.issue}."
        )
    return int(valid[-1])


def _std(values: np.ndarray, stream: str, what: str) -> float:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        raise SignalDataError(
            f"{stream!r} has too few values to estimate the spread of {what}; give "
            "it more history."
        )
    return float(np.std(values, ddof=1))


# -- baselines ---------------------------------------------------------------------


@register_step
class Persistence(ForecastStep):
    """Hold the last value: the target's last value before the issue time.

    A baseline that a forecaster has to beat. Its intervals widen with the square
    root of the lead time, from the spread of the training data's step-to-step
    changes.
    """

    kind: Literal["forecast_persistence"] = "forecast_persistence"

    @model_validator(mode="after")
    def _check_no_exog(self) -> Persistence:
        _no_exog(self)
        return self

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        history = window.history(stream)
        return {"sigma": _std(np.diff(history), stream, "its step-to-step changes")}

    def _predict(self, window, stream, state):
        last = _last_valid(window, stream)
        mean = np.full(len(window.future), window.history(stream)[last])
        leads = window.future - last
        return mean, state["sigma"] * np.sqrt(leads)


@register_step
class SeasonalNaive(ForecastStep):
    """Repeat the last season: each forecast value is the value one season earlier.

    With ``season="1D"``, the forecast for 14:00 tomorrow is the value at 14:00
    today, or the latest day before that with a value. A baseline that a
    forecaster has to beat. Its intervals widen with the square root of the
    number of seasons reached back, from the spread of the training data's
    season-to-season changes.
    """

    kind: Literal["forecast_seasonal_naive"] = "forecast_seasonal_naive"
    season: Duration = Field(
        "1D", description="Length of the season to repeat, e.g. 1D or 7D."
    )

    @model_validator(mode="after")
    def _check_no_exog(self) -> SeasonalNaive:
        _no_exog(self)
        return self

    def _period(self, window: _Window) -> int:
        return whole_steps(
            to_timedelta(self.season),
            window.step,
            what=f"season {self.season}",
            field="season",
        )

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        m = self._period(window)
        history = window.history(stream)
        if len(history) <= m:
            raise SignalDataError(
                f"seasonal_naive: {stream!r} needs more than one season "
                f"({self.season}) of history to fit on."
            )
        changes = history[m:] - history[:-m]
        return {"sigma": _std(changes, stream, "its season-to-season changes")}

    def _predict(self, window, stream, state):
        m = self._period(window)
        history = window.history(stream)
        mean = np.full(len(window.future), np.nan)
        seasons = np.ones(len(window.future))
        for i, position in enumerate(window.future):
            back = -(-(position - window.n_history + 1) // m)  # ceil: into the history
            source = position - back * m
            while source >= 0 and np.isnan(history[source]):
                source -= m
                back += 1
            if source < 0:
                raise SignalDataError(
                    f"seasonal_naive: {stream!r} has no value a whole number of "
                    f"seasons before {window.frame.index[position]}."
                )
            mean[i] = history[source]
            seasons[i] = back
        return mean, state["sigma"] * np.sqrt(seasons)


# -- structural (statsmodels) ------------------------------------------------------


@register_step
class Structural(ForecastStep):
    """Forecast with a structural time-series model (statsmodels' UnobservedComponents).

    The target is modelled as a level (``local_level``) or a level with a slope
    (``local_linear_trend``), plus ``seasonal`` cycles, an optional
    ``autoregressive`` term for slowly decaying departures, a linear effect of
    each ``exog`` stream, and noise. ``fit`` estimates the model's parameters by
    maximum likelihood on the training data and stores them as JSON;
    ``transform`` runs the Kalman filter over the new data's history with those
    parameters and forecasts from the issue time. The intervals come from the
    model's forecast variance. Missing target values in the history are fine;
    ``exog`` streams need a value at every time step.
    """

    kind: Literal["forecast_structural"] = "forecast_structural"
    level: Literal["local_level", "local_linear_trend"] = Field(
        "local_level", description="How the underlying level moves."
    )
    seasonal: tuple[CycleTerm, ...] = Field(
        (), description="Repeating cycles, e.g. [{period: 1D, harmonics: 2}]."
    )
    autoregressive: int = Field(
        0, ge=0, le=5, description="Order of an autoregressive term; 0 for none."
    )

    def _model(self, window: _Window, stream: str):
        cls = get_backend("statsmodels:UnobservedComponents")
        n = window.n_history
        exog = self._exog_matrix(window)
        if self.exog and np.isnan(exog[:n]).any():
            row, col = np.argwhere(np.isnan(exog[:n]))[0]
            raise SignalDataError(
                f"structural: exog stream {self.exog[col]!r} has no value at "
                f"{window.frame.index[row]}; fill it first (fs.impute)."
            )
        model = cls(
            window.history(stream),
            **structural_args(self.level, self.seasonal, window.step),
            autoregressive=self.autoregressive or None,
            exog=exog[:n] if self.exog else None,
        )
        return model, exog[window.future] if self.exog else None

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        if np.count_nonzero(~np.isnan(window.history(stream))) < 10:
            raise SignalDataError(
                f"structural: {stream!r} has fewer than 10 values to fit on."
            )
        model, _ = self._model(window, stream)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = model.fit(disp=False)
        converged = bool(result.mle_retvals.get("converged", True)) and not any(
            issubclass(w.category, ConvergenceWarning) for w in caught
        )
        return {
            "params": {
                name: float(value)
                for name, value in zip(model.param_names, result.params, strict=True)
            },
            "metrics": {"converged": converged, "aic": float(result.aic)},
        }

    def _predict(self, window, stream, state):
        model, exog_ahead = self._model(window, stream)
        params = state["params"]
        missing = sorted(set(model.param_names) - set(params))
        if missing:
            raise SignalConfigError(
                f"structural: the fitted state has no {missing}; refit the pipeline."
            )
        result = model.filter(np.array([params[n] for n in model.param_names]))
        forecast = result.get_forecast(len(window.future), exog=exog_ahead)
        return np.asarray(forecast.predicted_mean), np.asarray(forecast.se_mean)


# -- regression ---------------------------------------------------------------------


@register_step
class Regression(ForecastStep):
    """Forecast with a linear model of cycles, exog streams and lagged values.

    The inputs at each time are:

    - sine and cosine terms for each ``fourier`` cycle;
    - each ``exog`` stream at that time (known ahead);
    - the target itself ``lags`` earlier: the forecast is then made one step at a
      time, each step using the ones before it;
    - each ``past`` stream ``past_lags`` earlier. Those streams are known only up to
      the issue time, so every ``past_lags`` must be at least the horizon.

    ``fit`` learns one model per target on the training rows where the target and
    all its inputs have values, and stores the coefficients as JSON. The
    intervals are the training residuals' quantiles; they don't widen with lead
    time, so with ``lags`` they understate the uncertainty far ahead.
    """

    kind: Literal["forecast_regression"] = "forecast_regression"
    fourier: tuple[CycleTerm, ...] = Field(
        (), description="Cycles to model, e.g. [{period: 1D, harmonics: 3}]."
    )
    lags: tuple[Duration, ...] = Field(
        (), description="Also use the target this long before, e.g. [1h, 1D]."
    )
    past: tuple[str, ...] = Field(
        (),
        description=(
            "Streams known only up to the issue time, used past_lags earlier."
        ),
    )
    past_lags: tuple[Duration, ...] = Field(
        (),
        description="How long before to use each past stream; each at least horizon.",
    )
    model: str = Field(
        "sklearn:LinearRegression",
        description=(
            "A linear model key: least squares by default, or e.g. sklearn:Ridge "
            "with params {alpha: ...} when there are many inputs."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the model."
    )

    @field_validator("model")
    @classmethod
    def _linear_model(cls, value: str) -> str:
        check_model_key(value)
        if not is_linear_model(value):
            raise ValueError(f"regression needs a linear model, not {value!r}")
        return value

    _one_is_a_list = field_validator("lags", "past", "past_lags", mode="before")(
        one_is_a_list
    )

    @model_validator(mode="after")
    def _has_inputs(self) -> Regression:
        if not (self.fourier or self.exog or self.lags or self.past):
            raise ValueError("give fourier terms, exog, lags or past streams")
        if bool(self.past) != bool(self.past_lags):
            raise ValueError("past streams and past_lags go together")
        short = [
            lag
            for lag in self.past_lags
            if to_timedelta(lag) < to_timedelta(self.horizon)
        ]
        if short:
            raise ValueError(
                f"past_lags {short} are shorter than the horizon {self.horizon}: a "
                "past stream isn't known after the issue time"
            )
        overlap = sorted(set(self.past) & (set(self.streams) | set(self.exog)))
        if overlap:
            raise ValueError(f"{overlap} can't be past streams as well")
        return self

    def _shifts(self, window: _Window, lags: Sequence[str], field: str) -> list[int]:
        return [
            whole_steps(to_timedelta(lag), window.step, what=f"lag {lag}", field=field)
            for lag in lags
        ]

    def _fixed_inputs(self, window: _Window) -> np.ndarray:
        """Inputs that don't depend on the forecast: cycles, exog, past streams."""
        check_streams(window.frame, self.past, field="past")
        terms = [(to_timedelta(t.period), t.pairs(3)) for t in self.fourier]
        parts = [fourier_terms(window.frame.index, terms), self._exog_matrix(window)]
        for shift in self._shifts(window, self.past_lags, "past_lags"):
            for col in self.past:
                parts.append(
                    pd.Series(window.column(col)).shift(shift).to_numpy()[:, None]
                )
        return np.column_stack(parts)

    def _own_lags(self, y: np.ndarray, shifts: Sequence[int]) -> np.ndarray:
        if not shifts:
            return np.empty((len(y), 0))
        return np.column_stack([pd.Series(y).shift(s).to_numpy() for s in shifts])

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        n = window.n_history
        y = window.history(stream)
        shifts = self._shifts(window, self.lags, "lags")
        X = np.column_stack([self._fixed_inputs(window)[:n], self._own_lags(y, shifts)])
        rows = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        state = fit_linear(self.model, X[rows], y[rows], self.params)
        residuals = y[rows] - predict_linear(state, X[rows])
        state["residual_quantiles"] = {
            str(q): float(np.quantile(residuals, q)) for q in self.quantiles
        }
        return state

    def _predict(self, window, stream, state):
        fixed = self._fixed_inputs(window)
        shifts = self._shifts(window, self.lags, "lags")
        y = window.column(stream).copy()
        y[window.n_history :] = np.nan
        if not shifts:
            mean = predict_linear(state, fixed[window.future])
        else:
            mean = np.empty(len(window.future))
            for i, position in enumerate(window.future):
                lagged = [y[position - s] if position >= s else np.nan for s in shifts]
                row = np.concatenate([fixed[position], lagged])[None, :]
                mean[i] = predict_linear(state, row)[0]
                y[position] = mean[i]
        if np.isnan(mean).any():
            when = window.frame.index[window.future[np.isnan(mean)][0]]
            raise SignalDataError(
                f"regression: an input of the {stream!r} forecast is missing at {when} "
                "(a lagged value or a past stream). Fill gaps first (fs.impute)."
            )
        offsets = state["residual_quantiles"]
        return mean, {q: mean + offsets[str(q)] for q in self.quantiles}


__getattr__ = registered_lookup(__name__, "forecaster")
