"""Shared machinery for the built-in steps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.spec.models import StepSpec


def check_streams(
    sf: SignalFrame, streams: Sequence[str], field: str = "streams"
) -> None:
    """Raise a SignalConfigError naming any stream that the frame doesn't have."""
    missing = [s for s in streams if s not in sf]
    if missing:
        raise SignalConfigError(
            f"unknown stream(s) {missing}; the data has {sf.columns}.",
            field=field,
            value=missing,
        )


def remove_values(sf: SignalFrame, remove: pd.DataFrame, flag: Flag) -> SignalFrame:
    """Set values to NaN and add ``flag`` where ``remove`` is True and a value exists.

    Values that are already missing keep their flags unchanged.

    Args:
        sf: The frame.
        remove: Boolean frame; its columns are a subset of the frame's streams.
        flag: The reason, e.g. :attr:`Flag.OUTLIER`.
    """
    values = sf.values
    flags = sf.flags
    for col in remove.columns:
        hit = remove[col].to_numpy(dtype=bool) & values[col].notna().to_numpy()
        if not hit.any():
            continue
        col_values = values[col].to_numpy(copy=True)
        col_values[hit] = np.nan
        values[col] = col_values
        col_flags = flags[col].to_numpy(copy=True)
        col_flags[hit] |= FLAG_DTYPE(int(flag))
        flags[col] = col_flags
    return sf.evolve(values=values, flags=flags)


def regular_step(index: pd.DatetimeIndex, *, kind: str) -> pd.Timedelta:
    """Return the single spacing of a regular index, or raise.

    Raises:
        SignalDataError: If the index is irregular or has fewer than two points.
    """
    if len(index) < 2:
        raise SignalDataError(
            f"{kind} needs at least two timestamps on a regular grid; run align or "
            "resample first."
        )
    diffs = np.unique(np.diff(index.asi8))
    if len(diffs) != 1:
        raise SignalDataError(
            f"{kind} needs a regular time index (one spacing between all samples); "
            "run align or resample first."
        )
    return pd.Timedelta(int(diffs[0]), unit="ns")


def gaps_to_fill(
    values: np.ndarray,
    flags: np.ndarray,
    t: np.ndarray,
    max_gap_s: float | None,
) -> np.ndarray:
    """Return where an imputer may fill: gaps no longer than ``max_gap_s``.

    A gap is a run of missing values. Its length is the time between the valid
    sample before it and the valid sample after it, so a gap at the start or end
    of the data has no length and is filled only when ``max_gap_s`` is None.
    Values flagged ``EXCLUDED`` or ``NO_SAMPLE`` are never filled.

    Args:
        values: One stream's values.
        flags: Its flags.
        t: Timestamps in seconds (see :func:`seconds`).
        max_gap_s: Longest gap to fill, in seconds; None for any length.

    Returns:
        A boolean array, True where a value may be filled.
    """
    valid = ~np.isnan(values)
    excluded = (flags.astype("int64") & int(Flag.EXCLUDED | Flag.NO_SAMPLE)) != 0
    fill = ~valid & ~excluded
    if max_gap_s is None or not fill.any():
        return fill
    n = len(values)
    positions = np.arange(n)
    prev_idx = np.maximum.accumulate(np.where(valid, positions, -1))
    next_idx = np.minimum.accumulate(np.where(valid, positions, n)[::-1])[::-1]
    interior = (prev_idx >= 0) & (next_idx < n)
    span = np.full(n, np.inf)
    span[interior] = t[next_idx[interior]] - t[prev_idx[interior]]
    return fill & (span <= max_gap_s)


def fourier_terms(
    index: pd.DatetimeIndex, terms: Sequence[tuple[pd.Timedelta, int]]
) -> np.ndarray:
    """Return sine and cosine columns for each ``(period, harmonics)`` term.

    The phase is measured from the Unix epoch, so the same timestamp always gets
    the same values, whatever data it sits in.

    Returns:
        An array of shape ``(len(index), 2 * total_harmonics)``.
    """
    ns = index.asi8.astype("float64")
    columns = []
    for period, harmonics in terms:
        phase = 2 * np.pi * ns / float(period.value)
        for k in range(1, harmonics + 1):
            columns += [np.sin(k * phase), np.cos(k * phase)]
    return np.column_stack(columns) if columns else np.empty((len(index), 0))


def seconds(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the index as float seconds from its first timestamp."""
    ns = index.asi8
    return (
        (ns - ns[0]).astype("float64") / 1e9 if len(ns) else np.array([], dtype=float)
    )


class StreamStep(StepSpec):
    """A step that acts on named streams.

    The streams may be given positionally, as in ``Hampel("flow", window="1h")``.
    """

    streams: tuple[str, ...] = Field(
        min_length=1, description="The streams this step acts on."
    )

    def __init__(self, streams: str | Sequence[str] | None = None, /, **data: Any):
        """Accept ``streams`` positionally as well as by keyword."""
        if streams is not None:
            if "streams" in data:
                raise TypeError("streams given both positionally and by keyword")
            data["streams"] = streams
        super().__init__(**data)

    @field_validator("streams", mode="before")
    @classmethod
    def _one_stream_is_a_list(cls, value: Any) -> Any:
        return (value,) if isinstance(value, str) else value
