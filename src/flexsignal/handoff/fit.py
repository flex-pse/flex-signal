"""Prepare input and output data for a regressor.

``X`` and ``y`` must share one index, and time-series regressors need it
regular and gap-free. :func:`~flexsignal.handoff.fit_data` joins the input and
output frames on their timestamps, optionally resamples them, and deals with
gaps. Every step is logged, and the result carries a sha256 of the prepared data
(:func:`~flexsignal.handoff.data_hash`): the same data gives the same hash on
every run.

flex-signal stops at the prepared ``X, y``: pass them to a regressor yourself,
e.g. flex-pse's ``LinearRegressor().fit(X, y, input_units=fit.input_units,
output_units=fit.output_units)``. (Moved from flex-orchestrator,
unchanged.)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.frame import SignalFrame
from flexsignal.handoff.options import FitPrep

__all__ = ["FitData", "data_hash", "fit_data", "split_holdout"]


@dataclass
class FitData:
    """Data ready to fit.

    Attributes:
        X: The input columns.
        y: The one output column, as a one-column DataFrame.
        units: ``{column: units or None}`` for every column.
        steps: What was done, in order, e.g. ``{"step": "align", ...}``.
        data_hash: See :func:`data_hash`.
    """

    X: pd.DataFrame
    y: pd.DataFrame
    units: dict[str, str | None]
    steps: list[dict[str, Any]] = field(default_factory=list)
    data_hash: str = ""

    @property
    def input_units(self) -> dict[str, str | None]:
        """``{input column: units}``, as flex-pse's regressors take ``input_units``."""
        return {c: self.units.get(c) for c in self.X.columns}

    @property
    def output_units(self) -> str | None:
        """The output column's units, as flex-pse's regressors take ``output_units``."""
        return self.units.get(self.y.columns[0])


def _frame_and_units(item: Any) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if isinstance(item, SignalFrame):
        return item.values, item.units
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], pd.DataFrame)
        and isinstance(item[1], Mapping)
    ):
        frame, units = item
        return frame, {c: units.get(c) for c in frame.columns}
    if isinstance(item, pd.DataFrame):
        return item, dict.fromkeys(item.columns)
    raise SignalConfigError(
        "expected a SignalFrame, a DataFrame or (DataFrame, units), got "
        f"{type(item).__name__}."
    )


def _as_list(items: Any) -> list[Any]:
    if isinstance(items, SignalFrame | pd.DataFrame | tuple):
        return [items]
    return list(items)


def fit_data(
    X: Any,
    y: Any,
    prep: FitPrep | Mapping[str, Any] | None = None,
) -> FitData:
    """Join input and output data and prepare it for a regressor.

    Args:
        X: The inputs: a SignalFrame, a DataFrame, ``(DataFrame, units)``, or a
            list of them (one per source).
        y: The output, in the same forms; together they must give one column.
        prep: How to prepare it (:class:`~flexsignal.handoff.FitPrep` or a dict;
            default: inner join, no resampling, gaps are errors).

    Returns:
        A :class:`FitData`.

    Raises:
        SignalConfigError: If two sources give the same column, or ``y`` isn't
            one column.
        SignalDataError: If there are too many rows, gaps remain that ``prep``
            doesn't allow, or no rows are left.

    Example:
        >>> import pandas as pd
        >>> from flexsignal.handoff import fit_data
        >>> t = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
        >>> flow = pd.DataFrame({"flow": [1.0, 2.0, 3.0]}, index=t)
        >>> power = pd.DataFrame({"power": [2.0, 4.0, 6.0]}, index=t)
        >>> fit_data(flow, power).X.shape
        (3, 1)
    """
    if prep is None:
        prep = FitPrep()
    elif isinstance(prep, Mapping):
        prep = FitPrep.model_validate(prep)
    xs = [_frame_and_units(item) for item in _as_list(X)]
    ys = [_frame_and_units(item) for item in _as_list(y)]
    x_columns = [c for frame, _ in xs for c in frame.columns]
    y_columns = [c for frame, _ in ys for c in frame.columns]
    every = x_columns + y_columns
    repeated = sorted({c for c in every if every.count(c) > 1})
    if repeated:
        raise SignalConfigError(
            f"Columns {repeated} come from more than one selector; rename them."
        )
    if len(y_columns) != 1:
        raise SignalConfigError(f"y must give one column; it gives {y_columns}.")
    units = {c: u for _, part in xs + ys for c, u in part.items()}
    data = pd.concat(
        [frame for frame, _ in xs + ys], axis=1, join=prep.align, sort=True
    ).sort_index()
    steps: list[dict[str, Any]] = [
        {"step": "align", "how": prep.align, "rows": len(data)}
    ]
    if len(data) > prep.max_rows:
        raise SignalDataError(
            f"The selected data has {len(data)} rows, more than max_rows="
            f"{prep.max_rows}; narrow the window or resample."
        )
    if prep.resample is not None:
        data = data.resample(prep.resample.freq).agg(prep.resample.agg)
        steps.append(
            {
                "step": "resample",
                "freq": prep.resample.freq,
                "agg": prep.resample.agg,
                "rows": len(data),
            }
        )
    data = _gaps(data, prep, steps)
    if data.empty:
        raise SignalDataError("No rows are left to fit after preparing the data.")
    prepared = FitData(X=data[x_columns], y=data[y_columns], units=units, steps=steps)
    prepared.data_hash = data_hash(prepared.X, prepared.y)
    return prepared


def _gaps(
    data: pd.DataFrame, prep: FitPrep, steps: list[dict[str, Any]]
) -> pd.DataFrame:
    missing = int(data.isna().any(axis=1).sum())
    if not missing:
        return data
    if prep.gaps == "error":
        first = data.index[data.isna().any(axis=1)][0]
        raise SignalDataError(
            f"{missing} of {len(data)} rows have missing values (the first at "
            f"{first.isoformat()}). Set prep.gaps to 'drop', 'interpolate' or "
            "'longest_segment'."
        )
    if prep.gaps == "drop":
        data = data.dropna()
    elif prep.gaps == "interpolate":
        data = data.interpolate(method="time", limit=prep.limit, limit_area="inside")
        left = int(data.isna().any(axis=1).sum())
        if left:
            raise SignalDataError(
                f"{left} rows still have missing values after interpolating "
                f"(limit={prep.limit}, and the ends aren't extrapolated). Raise "
                "the limit, or use 'drop' or 'longest_segment'."
            )
    else:
        complete = ~data.isna().any(axis=1)
        segment = (complete != complete.shift()).cumsum()[complete]
        if segment.empty:
            raise SignalDataError("Every row has a missing value; there is no segment.")
        longest = segment.value_counts().idxmax()
        data = data[complete & (segment.reindex(data.index) == longest)]
    steps.append(
        {"step": "gaps", "how": prep.gaps, "rows_with_gaps": missing, "rows": len(data)}
    )
    return data


def data_hash(X: pd.DataFrame, y: pd.DataFrame) -> str:
    """Return the sha256 of prepared data, the same for the same data.

    Covers the timestamps, the column names and every value, exactly (floats are
    written in their shortest round-tripping form).
    """
    payload = {
        "time": [when.isoformat() for when in X.index],
        "X": {c: [float(v) for v in X[c].tolist()] for c in X.columns},
        "y": {c: [float(v) for v in y[c].tolist()] for c in y.columns},
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def split_holdout(
    X: pd.DataFrame, y: pd.DataFrame, holdout: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split off the last ``holdout`` (a pandas offset, e.g. ``"7D"``) as test rows.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.

    Raises:
        SignalConfigError: If the offset doesn't parse.
        SignalDataError: If either side would be empty.
    """
    try:
        cutoff = X.index.max() - pd.tseries.frequencies.to_offset(holdout)
    except ValueError as exc:
        raise SignalConfigError(f"Can't read holdout {holdout!r}: {exc}") from exc
    train = X.index <= cutoff
    if train.all() or not train.any():
        raise SignalDataError(
            f"Holding out {holdout!r} leaves no rows to "
            f"{'test' if train.all() else 'train'} on."
        )
    return X[train], y[train], X[~train], y[~train]
