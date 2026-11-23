"""Checks for the duration and timestamp strings that specs and conditions use.

A spec must give the same result on every machine and on every day it runs,
so a missing value (``"NaT"``) or a relative word such as
``"now"`` is refused here, even though pandas would accept them.
"""

from __future__ import annotations

import pandas as pd

RELATIVE_TIMESTAMPS = frozenset({"now", "today", "yesterday", "tomorrow"})
"""Words pandas reads as a time relative to the clock, refused in timestamps."""


def check_duration(
    value: str, *, allow_zero: bool = False, signed: bool = False
) -> pd.Timedelta:
    """Parse a duration string such as ``"15min"``, refusing a missing one.

    Args:
        value: The duration, as a pandas offset string.
        allow_zero: Accept ``"0s"``. Without it the duration must be positive.
        signed: Accept negative durations too (``allow_zero`` is then ignored).

    Returns:
        The duration.

    Raises:
        ValueError: If ``value`` is not a duration, is ``"NaT"``, or is out of
            range. Spec loading and condition checks turn it into a
            ``SignalConfigError`` with the field.
    """
    example = "'15min' or '-5min'" if signed else "'15min', '2h' or '1D'"
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        delta = None
    if delta is None or pd.isna(delta):
        raise ValueError(
            f"{value!r} is not a duration; use a pandas offset string such as "
            f"{example}"
        )
    if signed:
        return delta
    if allow_zero and delta < pd.Timedelta(0):
        raise ValueError(f"duration {value!r} must be 0 or more")
    if not allow_zero and delta <= pd.Timedelta(0):
        raise ValueError(f"duration {value!r} must be positive")
    return delta


def check_timestamp(value: str) -> pd.Timestamp:
    """Parse a fixed timestamp string, refusing ``"NaT"`` and ``"now"``-like words.

    Args:
        value: An ISO 8601 timestamp, e.g. ``"2025-01-01T00:00:00Z"``.

    Returns:
        The timestamp, as written (naive if it has no time zone).

    Raises:
        ValueError: If ``value`` is not a fixed timestamp.
    """
    if isinstance(value, str) and value.strip().lower() in RELATIVE_TIMESTAMPS:
        raise ValueError(
            f"{value!r} is relative to the clock, so the spec would give a different "
            "result each time it runs; write a fixed ISO 8601 timestamp, e.g. "
            "'2025-01-01T00:00:00Z'"
        )
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        ts = None
    if ts is None or pd.isna(ts):
        raise ValueError(
            f"{value!r} is not a timestamp; use ISO 8601, e.g. '2025-01-01T00:00:00Z'"
        )
    return ts
