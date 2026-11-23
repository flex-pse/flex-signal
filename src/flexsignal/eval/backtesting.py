"""Backtesting forecast steps: score them on the data's own past.

:func:`~flexsignal.eval.backtest` issues a pipeline's forecast at several times in
the data's past, each time with everything after the issue time hidden except the
streams known ahead (``exog``), and scores each forecast against the data. ``compare``
swaps other forecast steps into the same pipeline, so a model is scored against
baselines on exactly the same cleaned data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.eval.scores import metrics
from flexsignal.pipeline import DataLike, Pipeline, bind_inputs, resolve_spec
from flexsignal.spec import SignalSpec
from flexsignal.spec.models import to_timedelta
from flexsignal.steps.forecast import ForecastStep, quantile_column

__all__ = ["BacktestResult", "backtest"]

_HIDDEN = FLAG_DTYPE(int(Flag.MISSING | Flag.NO_SAMPLE))


@dataclass(frozen=True)
class BacktestResult:
    """What :func:`backtest` returns."""

    scores: pd.DataFrame
    """One row per method, stream and fold: ``issue_time``, ``n`` (values scored:
    an actual value and a forecast), ``missing`` (actual values the method left
    without a forecast), ``rmse``, ``mae``, ``mape``, and ``coverage`` (the share
    of values between the lowest and highest quantile; NaN without quantiles)."""
    forecasts: pd.DataFrame
    """Every forecast value: ``method``, ``stream``, ``fold``, ``issue_time``,
    ``time``, ``lead`` (time after the issue time), ``actual``, ``forecast``, and
    one column per quantile (``q10``, ...)."""
    baseline: str | None
    """The method :meth:`summary` measures skill against."""

    def summary(self) -> pd.DataFrame:
        """Return each method's scores averaged over the folds, with skill.

        ``skill`` is ``1 - rmse / baseline rmse`` per stream: above 0 beats the
        baseline, 1 is perfect. NaN without a baseline.
        """
        table = (
            self.scores.groupby(["method", "stream"], sort=False)[
                ["rmse", "mae", "mape", "coverage"]
            ]
            .mean()
            .reset_index()
        )
        table["skill"] = np.nan
        if self.baseline is not None:
            base = table[table["method"] == self.baseline].set_index("stream")["rmse"]
            table["skill"] = 1.0 - table["rmse"] / table["stream"].map(base)
        return table

    def to_dict(self) -> dict[str, Any]:
        """Return the scores and summary as JSON-able lists of records."""

        def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            frame = frame.copy()
            for col in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    frame[col] = frame[col].map(pd.Timestamp.isoformat)
            out = frame.astype(object).where(frame.notna(), None)
            return [
                {k: v.item() if isinstance(v, np.generic) else v for k, v in r.items()}
                for r in out.to_dict(orient="records")
            ]

        return {
            "baseline": self.baseline,
            "scores": records(self.scores),
            "summary": records(self.summary()),
        }


def _duration(value: str, field: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        delta = None
    if delta is None or delta <= pd.Timedelta(0):
        raise SignalConfigError(
            f"{field} {value!r} isn't a positive duration; use e.g. '1D' or '12h'.",
            field=field,
        )
    return delta


def _forecast_step(spec: SignalSpec) -> int:
    found = [i for i, s in enumerate(spec.steps) if isinstance(s, ForecastStep)]
    if len(found) != 1:
        raise SignalConfigError(
            f"backtest needs a spec with exactly one forecast step; it has "
            f"{len(found)}.",
            field="steps",
        )
    return found[0]


def _label(step: ForecastStep) -> str:
    return step.name or step.kind


def _issue_times(
    sf: SignalFrame,
    step: ForecastStep,
    folds: int,
    every: pd.Timedelta,
) -> list[pd.Timestamp]:
    valid = sf.values[list(step.streams)].notna().any(axis=1).to_numpy()
    if not valid.any():
        raise SignalDataError(
            f"backtest: the targets {list(step.streams)} have no values."
        )
    last = sf.index[np.flatnonzero(valid)[-1]] - to_timedelta(step.horizon)
    times = [last - k * every for k in range(folds)][::-1]
    if times[0] <= sf.index[0]:
        raise SignalDataError(
            f"backtest: {folds} folds every {every} need data from before "
            f"{times[0]}, but the data starts at {sf.index[0]}. Use fewer folds, a "
            "shorter every, or more data."
        )
    return times


def _fold_data(
    sf: SignalFrame,
    issue: pd.Timestamp,
    horizon: pd.Timedelta,
    known_ahead: set[str],
    train: pd.Timedelta | None,
) -> SignalFrame:
    keep = sf.index <= issue + horizon
    if train is not None:
        keep &= sf.index > issue - train
    values = sf.values.loc[keep].copy()
    flags = sf.flags.loc[keep].copy()
    after = values.index > issue
    for col in values.columns:
        if col in known_ahead:
            continue
        values.loc[after, col] = np.nan
        flags.loc[after, col] = _HIDDEN
    return SignalFrame(values, flags, sf.meta)


def _suffix(stream: str, q: float) -> str:
    """Return a quantile's column name in the forecasts table, e.g. ``q10``."""
    return quantile_column(stream, q).rsplit("__", 1)[1]


def backtest(
    spec: SignalSpec | Mapping[str, Any] | str | Path,
    data: DataLike = None,
    *,
    folds: int = 3,
    every: str | None = None,
    train: str | None = None,
    compare: Sequence[ForecastStep] = (),
    baseline: str | None = None,
    base_dir: str | Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    plugins: str | Sequence[str] = (),
) -> BacktestResult:
    """Score a pipeline's forecast step on the data's past, fold by fold.

    The folds are blocked and in time order. The last fold's forecast ends at the
    last value of the targets; each earlier fold is issued ``every`` before the
    next. For each fold:

    1. every stream except the forecast step's ``exog`` is hidden after the issue
       time (flagged ``MISSING`` and ``NO_SAMPLE``, so no step fills it);
    2. the whole pipeline is fitted and run, with the forecast issued at that
       time;
    3. the forecast is scored against the data that was hidden.

    Args:
        spec: A SignalSpec, a spec dict, JSON text or a ``.json`` path, with exactly
            one forecast step.
        data: The data, as for :func:`flexsignal.run`.
        folds: How many forecasts to issue.
        every: Time between issue times (default: the horizon, so the forecasts
            don't overlap).
        train: Train on only this much data before each issue time (default: all
            of it).
        compare: Other forecast steps to score, each swapped in for the spec's
            forecast step. They must forecast the same streams over the same
            horizon. Their ``issue_time`` is set by the backtest.
        baseline: The method :meth:`BacktestResult.summary` measures skill against,
            by its name (``name``, or else ``kind``). Default: the first of
            ``compare``.
        base_dir: Directory for relative paths in the spec. Default: the spec
            file's directory for a path, else the working directory (as
            :func:`flexsignal.run`).
        progress: Called as ``progress(fraction, message)`` after each fold.
        plugins: Modules to import first, so the methods they register are
            available (see :func:`flexsignal.load_plugins`).

    Returns:
        A :class:`BacktestResult`.

    Raises:
        SignalConfigError: If the spec has no forecast step or several, a
            ``compare`` step doesn't match it, or two methods share a name.
        SignalDataError: If the data is too short for the folds.
    """
    resolved, root = resolve_spec(spec, base_dir, plugins)
    position = _forecast_step(resolved)
    main = resolved.steps[position]
    if folds < 1:
        raise SignalConfigError("folds must be at least 1.", field="folds")
    methods = [main, *compare]
    for other in compare:
        if not isinstance(other, ForecastStep):
            raise SignalConfigError(
                f"compare takes forecast steps, not {type(other).__name__}.",
                field="compare",
            )
        if other.streams != main.streams or other.horizon != main.horizon:
            raise SignalConfigError(
                f"compare step {_label(other)!r} forecasts {list(other.streams)} over "
                f"{other.horizon}; it must match the spec's {list(main.streams)} over "
                f"{main.horizon}.",
                field="compare",
            )
    labels = [_label(m) for m in methods]
    if len(set(labels)) != len(labels):
        raise SignalConfigError(
            f"two methods share a name in {labels}; give each a name=.",
            field="compare",
        )
    if baseline is None:
        baseline = labels[1] if len(labels) > 1 else None
    elif baseline not in labels:
        raise SignalConfigError(
            f"baseline {baseline!r} isn't one of the methods {labels}.",
            field="baseline",
        )

    data_sf = bind_inputs(resolved, data, base_dir=root)
    # Each fold passes one bound frame, writes no files and sets its issue time.
    single = resolved.model_copy(update={"inputs": (), "outputs": ()})
    horizon = to_timedelta(main.horizon)
    step_every = _duration(every, "every") if every is not None else horizon
    span = _duration(train, "train") if train is not None else None
    # The data as the forecast step sees it (cleaned, on its grid), from all of
    # the data: it sets the issue times and is what each forecast is scored on.
    before = single.model_copy(update={"steps": single.steps[:position]})
    reference = (
        Pipeline.from_spec(before).fit_transform(data_sf)[1] if position else data_sf
    )
    times = _issue_times(reference, main, folds, step_every)
    # Streams known ahead stay visible after each issue time: the exog of every
    # method compared. A step can't take its own target as exog, and compared
    # steps share the targets, so no target leaks.
    known_ahead = {s for m in methods for s in m.exog}

    score_rows: list[dict[str, Any]] = []
    forecast_rows: list[pd.DataFrame] = []
    total = len(times) * len(methods)
    done = 0
    for fold, issue in enumerate(times):
        fold_sf = _fold_data(data_sf, issue, horizon, known_ahead, span)
        for label, method in zip(labels, methods, strict=True):
            step = method.model_copy(update={"issue_time": issue.isoformat()})
            steps = list(single.steps)
            steps[position] = step
            pipe = Pipeline.from_spec(single.model_copy(update={"steps": tuple(steps)}))
            _, result = pipe.fit_transform(fold_sf)
            for stream in main.streams:
                rows = result.has_flag(Flag.FORECAST)[stream].to_numpy() & (
                    result.index > issue
                )
                times_ahead = result.index[rows]
                predicted = result.values[stream].to_numpy()[rows]
                actual = (
                    reference.values[stream].reindex(times_ahead).to_numpy()
                    if len(times_ahead)
                    else np.array([])
                )
                frame = pd.DataFrame(
                    {
                        "method": label,
                        "stream": stream,
                        "fold": fold,
                        "issue_time": issue,
                        "time": times_ahead,
                        "lead": times_ahead - issue,
                        "actual": actual,
                        "forecast": predicted,
                    }
                )
                for q in step.quantiles:
                    column = quantile_column(stream, q)
                    frame[_suffix(stream, q)] = result.values[column].to_numpy()[rows]
                forecast_rows.append(frame)
                known = ~np.isnan(actual)
                scored = known & ~np.isnan(predicted)
                row = {
                    "method": label,
                    "stream": stream,
                    "fold": fold,
                    "issue_time": issue,
                    "n": int(scored.sum()),
                    "missing": int((known & ~scored).sum()),
                    "rmse": np.nan,
                    "mae": np.nan,
                    "mape": np.nan,
                    "coverage": np.nan,
                }
                if scored.any():
                    row.update(
                        metrics(pd.Series(actual[scored]), pd.Series(predicted[scored]))
                    )
                if len(step.quantiles) >= 2 and scored.any():
                    low = frame[_suffix(stream, step.quantiles[0])].to_numpy()
                    high = frame[_suffix(stream, step.quantiles[-1])].to_numpy()
                    inside = (actual >= low) & (actual <= high)
                    row["coverage"] = float(inside[scored].mean())
                score_rows.append(row)
            done += 1
            if progress is not None:
                progress(done / total, f"fold {fold}: {label}")
    return BacktestResult(
        scores=pd.DataFrame(score_rows),
        forecasts=pd.concat(forecast_rows, ignore_index=True),
        baseline=baseline,
    )
