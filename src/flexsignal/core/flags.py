"""Per-sample quality flags.

Every value in a :class:`~flexsignal.core.frame.SignalFrame` has a matching
``uint16`` bitmask that records what happened to it. Cleaning never deletes data:
it sets the value to NaN and sets a flag bit, so a consumer can always ask which
values are real, which were removed and why, and which were filled in.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable

import numpy as np

FLAG_DTYPE = np.uint16
"""numpy dtype of a flags column."""


class Flag(enum.IntFlag):
    """Quality-flag bits; a sample's flags are the OR of every bit that applies."""

    RAW = 0
    MISSING = 1 << 0
    OUT_OF_RANGE = 1 << 1
    RATE_OF_CHANGE = 1 << 2
    OUTLIER = 1 << 3
    STUCK = 1 << 4
    MANUAL = 1 << 5
    EXCLUDED = 1 << 6
    CORRECTED = 1 << 7
    IMPUTED_SHORT = 1 << 8
    IMPUTED_MODEL = 1 << 9
    FORECAST = 1 << 10
    DERIVED = 1 << 11
    RESAMPLED = 1 << 12
    NO_SAMPLE = 1 << 13


FLAG_DESCRIPTIONS: dict[Flag, tuple[str, str]] = {
    Flag.MISSING: (
        "No value: missing (NaN) in the input, or no sample existed at that time.",
        "input data, align, resample, shift",
    ),
    Flag.OUT_OF_RANGE: ("The value was outside its allowed range.", "range"),
    Flag.RATE_OF_CHANGE: (
        "The value changed faster than physically allowed.",
        "rate_of_change",
    ),
    Flag.OUTLIER: (
        "The value was a statistical outlier.",
        "hampel, tukey",
    ),
    Flag.STUCK: (
        "The value was part of a flat-lined (stuck sensor) run.",
        "stuck",
    ),
    Flag.MANUAL: ("A user-written rule marked the value invalid.", "expression"),
    Flag.EXCLUDED: (
        "Deliberately left out by a selection rule; not bad data, and never filled.",
        "mask",
    ),
    Flag.CORRECTED: ("The value was corrected by a model.", "calibrate (M2)"),
    Flag.IMPUTED_SHORT: (
        "A short gap was filled by interpolation.",
        "interpolate",
    ),
    Flag.IMPUTED_MODEL: ("A gap was filled by a model.", "model imputers (M2)"),
    Flag.FORECAST: ("The value is a forecast, not a measurement.", "forecast (M4)"),
    Flag.DERIVED: ("The stream was computed from other streams.", "combine"),
    Flag.RESAMPLED: (
        "The value was aggregated or moved onto a new time grid.",
        "align, resample",
    ),
    Flag.NO_SAMPLE: (
        "The stream has no sample at this time; the row exists only because "
        "another input does. Always set with MISSING. Align, resample and "
        "interpolate skip these rows.",
        "joining several inputs",
    ),
}
"""Flag → (what it means, which steps set it). Rendered into the docs flag table."""

QUALITY_FLAGS = (
    Flag.OUT_OF_RANGE | Flag.RATE_OF_CHANGE | Flag.OUTLIER | Flag.STUCK | Flag.MANUAL
)
"""Bits set when a cleaner removes a value."""

ALL_BITS: tuple[Flag, ...] = tuple(f for f in Flag if f.value)
"""Every non-zero flag bit, lowest first."""


def flag_names(value: int) -> list[str]:
    """Return the names of the bits set in ``value``, lowest bit first.

    Args:
        value: A flags bitmask.

    Returns:
        The flag names, e.g. ``["MISSING", "RESAMPLED"]``; empty for a raw value.
    """
    return [f.name for f in ALL_BITS if value & f.value]


def parse_flags(names: str | Iterable[str]) -> Flag:
    """Combine flag names into one bitmask.

    Args:
        names: A flag name such as ``"OUTLIER"``, or several names.

    Returns:
        The combined :class:`Flag`.

    Raises:
        ValueError: If a name is not a flag.
    """
    if isinstance(names, str):
        names = [names]
    result = Flag.RAW
    for name in names:
        try:
            result |= Flag[name]
        except KeyError:
            known = ", ".join(f.name for f in ALL_BITS)
            raise ValueError(f"unknown flag {name!r}; known flags: {known}") from None
    return result
