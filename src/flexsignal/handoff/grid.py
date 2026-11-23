"""Line timeseries data up with a model's time steps.

A flex-pse model runs on the facility's wall clock: its time points are naive
local times, each the start of a step. :func:`~flexsignal.handoff.to_grid`
turns timeseries data into one value per time point:

1. its UTC timestamps go to the facility's ``timezone`` and lose it;
2. the data is shifted by ``data_years`` and ``offset`` onto the model's
   horizon;
3. it is resampled onto the model's time steps and its gaps filled;
4. or, with ``reduce``, the horizon's data becomes one number.

Only data inside :func:`~flexsignal.handoff.data_window` is used: the horizon,
plus a day either side, so a gap at either end can be interpolated from the
samples beyond it.

This is flex-orchestrator's ``model_inputs`` alignment, moved here so a plain
script can use it; its behaviour, error messages and hashes are
unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.handoff.options import GridOptions, GridResample

__all__ = ["GridValues", "data_window", "to_grid", "values_hash"]

MARGIN = pd.Timedelta(days=1)
"""How far past each end of the horizon data is read, so end gaps can be filled."""

_ESTIMATED = int(
    Flag.IMPUTED_SHORT | Flag.IMPUTED_MODEL | Flag.CORRECTED | Flag.FORECAST
)
_RESAMPLED = int(Flag.RESAMPLED)


@dataclass
class GridValues:
    """What :func:`to_grid` gives: one value per time step, or one number, per column.

    Attributes:
        values: ``{column: [one float per time step]}``, or ``{column: float}``
            with ``reduce``.
        units: ``{column: units or None}``, written for flex-pse (``m^3``, not
            ``m**3``).
        filled: ``{column: time steps that had no value before the gap rule}``;
            empty with ``reduce``.
        estimated: ``{column: time steps whose value draws on samples flex-signal
            filled, corrected or forecast}`` (flags ``IMPUTED_SHORT``,
            ``IMPUTED_MODEL``, ``CORRECTED``, ``FORECAST``). Zero for data without
            flags.
        resampled: ``{column: time steps whose value draws on samples flex-signal
            put on a new time grid}`` (flag ``RESAMPLED``, from ``align`` or
            ``resample``). Not counted in ``estimated``: a mean of real samples is
            a measurement, but after upsampling (``align`` to a finer grid) most
            values are interpolated between readings. Zero for data without flags.
        window: The data read, in the facility's time: ``{"start", "end"}``, or
            None for a model without a time axis.
        timezone: The facility's time zone.
    """

    values: dict[str, list[float] | float]
    units: dict[str, str | None]
    filled: dict[str, int] = field(default_factory=dict)
    estimated: dict[str, int] = field(default_factory=dict)
    window: dict[str, str] | None = None
    timezone: str = "UTC"
    resampled: dict[str, int] = field(default_factory=dict)

    def to_overrides(
        self, rename: Mapping[str, str] | None = None, *, strict: bool = False
    ) -> dict[str, Any]:
        """Return ``{model path: value}``, as ``flexpse.apply_values`` takes it.

        Args:
            rename: Column → model path, e.g. ``{"flow": "plant.pump.flow_in"}``.
                Columns not listed keep their name.
            strict: Require every column to be in ``rename``, so a column left
                out can't be applied to a model path that happens to match its
                name.

        Returns:
            Each value as is, or ``{"value": ..., "units": ...}`` when the column
            has units.

        Raises:
            SignalConfigError: With ``strict``, if a column isn't in ``rename``.
        """
        rename = dict(rename or {})
        if strict:
            missing = [c for c in self.values if c not in rename]
            if missing:
                raise SignalConfigError(
                    f"columns {missing} have no model path in rename; add them, or "
                    "leave them out of the data given to to_grid.",
                    field="rename",
                )
        out: dict[str, Any] = {}
        for column, value in self.values.items():
            unit = self.units.get(column)
            out[rename.get(column, column)] = (
                {"value": value, "units": unit} if unit else value
            )
        return out

    def data_hash(self, rename: Mapping[str, str] | None = None) -> str:
        """Return the sha256 of :meth:`to_overrides`, as flex-orchestrator records."""
        return values_hash(self.to_overrides(rename))


def values_hash(values: Mapping[str, Any]) -> str:
    """Return ``"sha256:..."`` of ``{path: value}``, sorted, compact JSON."""
    text = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _step(grid: pd.DatetimeIndex) -> pd.Timedelta:
    if grid.freq is not None:
        return pd.Timedelta(grid.freq)
    if len(grid) < 2:
        raise SignalConfigError("The model's time grid has one point and no freq.")
    return grid[1] - grid[0]


def _years(options: GridOptions, start: pd.Timestamp) -> int:
    """How many years a horizon's data year is from its modeled year."""
    if not options.data_years or start.year not in options.data_years:
        return 0
    return options.data_years[start.year] - start.year


def _utc(local: pd.Timestamp, zone: str) -> pd.Timestamp:
    """A naive local time as UTC; a time DST skips or repeats takes the later."""
    stamp = pd.Timestamp(local).tz_localize(
        zone, ambiguous=False, nonexistent="shift_forward"
    )
    return stamp.tz_convert("UTC")


def _direction(options: GridOptions) -> str:
    return (options.resample or GridResample()).window


def _horizon(
    grid: pd.DatetimeIndex, options: GridOptions
) -> tuple[pd.Timestamp, pd.Timestamp, int, pd.Timedelta, pd.Timestamp, pd.Timestamp]:
    step = _step(grid)
    start, end = grid[0], grid[-1] + step
    years = _years(options, start)
    shift = pd.Timedelta(options.offset) if options.offset else pd.Timedelta(0)
    if years:
        start_in_data = start + pd.DateOffset(years=years)
        end_in_data = end + pd.DateOffset(years=years)
    else:
        start_in_data, end_in_data = start, end
    return start, end, years, shift, start_in_data + shift, end_in_data + shift


def data_window(
    grid: pd.DatetimeIndex | None, options: GridOptions | None = None
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the UTC window of data :func:`to_grid` uses for a horizon.

    The horizon, moved by ``data_years`` and ``offset``, plus ``MARGIN`` (a day) on
    each side (none with ``reduce``), converted from the facility's time to UTC.
    Both ends are included.

    Args:
        grid: The model's time points (naive local time), or None for a model
            without a time axis.
        options: How the data is lined up (default: :class:`GridOptions`).

    Returns:
        ``(start, end)`` as tz-aware UTC timestamps, or None for no time axis
        (all the data is used).
    """
    options = options or GridOptions()
    if grid is None:
        return None
    _, _, _, _, first, last = _horizon(grid, options)
    margin = pd.Timedelta(0) if options.reduce else MARGIN
    return _utc(first - margin, options.timezone), _utc(last + margin, options.timezone)


def _parts(
    data: Any,
) -> tuple[pd.DataFrame, dict[str, str | None], pd.DataFrame | None]:
    """Values, units and (if any) flags of a SignalFrame, DataFrame or (df, units)."""
    if isinstance(data, SignalFrame):
        return data.values, data.units, data.flags
    if (
        isinstance(data, tuple)
        and len(data) == 2
        and isinstance(data[0], pd.DataFrame)
        and isinstance(data[1], Mapping)
    ):
        frame, units = data
    elif isinstance(data, pd.DataFrame):
        frame, units = data, {}
    else:
        raise SignalConfigError(
            "expected a SignalFrame, a DataFrame or (DataFrame, units), got "
            f"{type(data).__name__}."
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise SignalDataError("the data's index must be a DatetimeIndex.")
    if frame.index.tz is None:
        # Naive timestamps follow the usual rule (local time, with a warning).
        sf = SignalFrame.from_frame(frame)
        return sf.values, {c: units.get(c) for c in sf.columns}, None
    return frame, {c: units.get(c) for c in frame.columns}, None


def to_grid(
    data: Any,
    grid: pd.DatetimeIndex | None,
    options: GridOptions | Mapping[str, Any] | None = None,
    *,
    columns: Sequence[str] | None = None,
) -> GridValues:
    """Line data up with a model's time steps: one value per step, per column.

    Args:
        data: A :class:`~flexsignal.core.SignalFrame`, a DataFrame with a
            tz-aware index, or ``(DataFrame, units)``.
        grid: The model's time points as naive local times, e.g. from
            :func:`flexsignal.integrations.flexpse.time_grid`; None for a model
            without a time axis, which takes only ``reduce``.
        options: How to line the data up (:class:`GridOptions` or a dict).
        columns: The columns to use (default: all).

    Returns:
        A :class:`GridValues`.

    Raises:
        SignalConfigError: If the options don't fit the grid, e.g. a series for a
            model without a time axis.
        SignalDataError: If there is no data in the window, a step is left
            without a value after the gap rule, or summed data can't be split
            over whole steps.

    Example:
        >>> import pandas as pd
        >>> from flexsignal.handoff import to_grid
        >>> hours = pd.date_range("2025-07-08", periods=4, freq="h", tz="UTC")
        >>> data = pd.DataFrame({"flow": [1.0, 2.0, 3.0, 4.0]}, index=hours)
        >>> grid = pd.date_range("2025-07-08", periods=4, freq="30min")
        >>> to_grid(data, grid).values["flow"]
        [1.0, 1.5, 2.0, 2.5]
    """
    if options is None:
        options = GridOptions()
    elif isinstance(options, Mapping):
        options = GridOptions.model_validate(options)
    frame, units, flags = _parts(data)
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise SignalConfigError(
                f"unknown column(s) {missing}; the data has {list(frame.columns)}."
            )
        frame = frame[list(columns)]
        flags = flags[list(columns)] if flags is not None else None
    if grid is not None:
        grid = pd.DatetimeIndex(grid)
        if grid.tz is not None:
            raise SignalConfigError(
                "the model's time grid must be naive local time, as flex-pse's is."
            )
    window = data_window(grid, options)
    if window is not None:
        inside = (frame.index >= window[0]) & (frame.index <= window[1])
        frame = frame[inside]
        flags = flags[inside] if flags is not None else None
    if frame.empty:
        raise SignalDataError(
            "the data has no samples in the window the model's horizon reads"
            + (f" ({window[0]} to {window[1]} UTC)." if window else ".")
        )

    unit_out = {c: (u.replace("**", "^") if u else None) for c, u in units.items()}
    unit_out = {c: unit_out.get(c) for c in frame.columns}
    if grid is None:
        if options.reduce is None:
            raise SignalConfigError(
                "the data gives a series, but the model has no time axis; set reduce."
            )
        local = frame.index.tz_convert(options.timezone).tz_localize(None)
        frame = frame.set_axis(local)
        values = {c: _reduce(frame[c], options, None, None) for c in frame.columns}
        return GridValues(values, unit_out, timezone=options.timezone)

    start, end, years, shift, first, last = _horizon(grid, options)
    local = frame.index.tz_convert(options.timezone).tz_localize(None) - shift
    index = local - pd.DateOffset(years=years) if years else local
    frame = frame.set_axis(index)
    if flags is not None:
        flags = flags.set_axis(index)
    step = _step(grid)
    # A backward step reads the interval ending at it, a step earlier.
    lag = step if _direction(options) == "backward" else pd.Timedelta(0)
    record = {"start": str(first - lag), "end": str(last - lag)}

    values: dict[str, Any] = {}
    filled: dict[str, int] = {}
    estimated: dict[str, int] = {}
    resampled: dict[str, int] = {}
    for column in frame.columns:
        series = frame[column]
        if options.reduce is not None:
            values[column] = _reduce(series, options, start, end)
            continue
        values[column], filled[column] = _series(series, options, grid)
        for counts, bits in ((estimated, _ESTIMATED), (resampled, _RESAMPLED)):
            if flags is None:
                counts[column] = 0
                continue
            marked = ((flags[column].to_numpy() & bits) != 0) & (
                series.notna().to_numpy()
            )
            counts[column] = _count_estimated(
                pd.Series(marked.astype(float), index=series.index, name=column),
                options,
                grid,
            )
    return GridValues(
        values,
        unit_out,
        filled,
        estimated,
        record,
        timezone=options.timezone,
        resampled=resampled,
    )


def _bins(series: pd.Series, options: GridOptions, grid: pd.DatetimeIndex):
    step = _step(grid)
    backward = _direction(options) == "backward"
    # forward: [t, t + dt) labelled t; backward: (t - dt, t] labelled t.
    side = "right" if backward else "left"
    return series.resample(step, origin=grid[0], closed=side, label=side)


def _series(
    series: pd.Series, options: GridOptions, grid: pd.DatetimeIndex
) -> tuple[list[float], int]:
    step = _step(grid)
    agg = (options.resample or GridResample()).agg
    backward = _direction(options) == "backward"
    if agg == "sum":
        series = _split(series, step, backward)
    bins = _bins(series, options, grid)
    binned = bins.sum(min_count=1) if agg == "sum" else getattr(bins, agg)()
    # The bins share the grid's origin and step, so the union stays regular.
    extended = binned.reindex(binned.index.union(grid))
    missing = int(extended.reindex(grid).isna().sum())
    if options.gaps == "interpolate":
        extended = extended.interpolate(
            method="time", limit=options.limit, limit_area="inside"
        )
    elif options.gaps == "ffill":
        extended = extended.ffill(limit=options.limit)
    elif options.gaps == "zero":
        extended = extended.fillna(0.0)
    aligned = extended.reindex(grid)
    left = aligned.index[aligned.isna()]
    if len(left):
        shown = ", ".join(str(t) for t in left[:3]) + (", ..." if len(left) > 3 else "")
        raise SignalDataError(
            f"{series.name!r} has no value at {len(left)} of the model's {len(grid)} "
            f"time steps ({shown}) after gaps: {options.gaps}"
            + (f" with limit {options.limit}" if options.limit else "")
            + ". Read a wider window, or fill the gaps another way.",
            field=str(series.name),
        )
    return [float(v) for v in aligned], missing


def _count_estimated(
    marked: pd.Series, options: GridOptions, grid: pd.DatetimeIndex
) -> int:
    """Time steps whose bin holds at least one filled or corrected sample."""
    agg = (options.resample or GridResample()).agg
    if agg == "sum":
        marked = _split(marked, _step(grid), _direction(options) == "backward")
    hit = _bins(marked, options, grid).max()
    return int((hit.reindex(grid).fillna(0.0) > 0).sum())


def _split(series: pd.Series, step: pd.Timedelta, backward: bool) -> pd.Series:
    """Spread amounts coarser than ``step`` evenly over the steps they cover.

    A sample stands for the interval that starts at it (forward) or ends at it
    (backward); its amount is split into one equal part per step of that
    interval, so a sum over any steps keeps the total.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return series
    spacing = pd.Series(clean.index).diff().median()
    if spacing <= step:
        return series
    parts, rest = divmod(spacing, step)
    if rest:
        raise SignalDataError(
            f"{series.name!r} sums data every {spacing} onto steps of {step}; its "
            "amounts can only be split over a whole number of steps.",
            field=str(series.name),
        )
    sign = -1 if backward else 1
    pieces = [
        pd.Series(clean.to_numpy() / parts, clean.index + sign * k * step)
        for k in range(int(parts))
    ]
    return pd.concat(pieces).sort_index().rename(series.name)


def _reduce(series: pd.Series, options: GridOptions, start: Any, end: Any) -> float:
    if start is not None:
        series = series[(series.index >= start) & (series.index < end)]
    series = series.dropna()
    if series.empty:
        raise SignalDataError(
            f"{series.name!r} has no values over the model's horizon to reduce.",
            field=str(series.name),
        )
    if options.reduce == "first":
        return float(series.iloc[0])
    if options.reduce == "last":
        return float(series.iloc[-1])
    return float(getattr(series, options.reduce or "mean")())
