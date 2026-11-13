"""Imputation steps: fill missing values and flag them as filled.

M1 has gap-length-aware interpolation. Model-based imputers (Kalman, regression,
PyPOTS) and the tiered imputer arrive in M2 and M5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from pydantic import Field

from flexsignal.backends import get_backend
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.spec.models import Duration, to_timedelta
from flexsignal.steps._base import StreamStep, check_streams, seconds

__all__ = ["Interpolate"]

_METHOD_BACKENDS = {
    "linear": "numpy:interp",
    "nearest": "scipy:interp1d",
    "previous": "scipy:interp1d",
    "pchip": "scipy:PchipInterpolator",
    "akima": "scipy:Akima1DInterpolator",
}


def _interpolator(method: str, x: np.ndarray, y: np.ndarray):
    backend = get_backend(_METHOD_BACKENDS[method])
    if method == "linear":
        return lambda xi: backend(xi, x, y)
    if method in ("nearest", "previous"):
        return backend(x, y, kind=method, assume_sorted=True)
    return backend(x, y)


@register_step
class Interpolate(StreamStep):
    """Fill short interior gaps by interpolating between the valid values either side.

    A gap is a run of missing values. Its length is the time between the valid
    sample before it and the valid sample after it. Only gaps no longer than
    ``max_gap`` are filled: on 5-minute data, ``max_gap="15min"`` fills up to two
    missing samples in a row. Gaps at the start or end of the data, and values
    flagged ``EXCLUDED`` or ``NO_SAMPLE``, are never filled. Filled values keep
    the flags that explain why they were missing, and gain ``IMPUTED_SHORT``.

    Methods use numpy and scipy:

    - ``linear``: :func:`numpy.interp`;
    - ``nearest`` and ``previous``: :class:`scipy.interpolate.interp1d`;
    - ``pchip``: :class:`scipy.interpolate.PchipInterpolator`, shape-preserving
      with no overshoot;
    - ``akima``: :class:`scipy.interpolate.Akima1DInterpolator`.
    """

    kind: Literal["interpolate"] = "interpolate"
    max_gap: Duration = Field(
        description=(
            "Longest gap to fill, measured between the valid samples either side, e.g. "
            "15min."
        )
    )
    method: Literal["linear", "nearest", "previous", "pchip", "akima"] = Field(
        "linear", description="Interpolation method."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill short gaps (see the class docstring)."""
        check_streams(sf, self.streams)
        max_gap_s = to_timedelta(self.max_gap).total_seconds()
        t = seconds(sf.index)
        values, flags = sf.values, sf.flags
        for col in self.streams:
            arr = values[col].to_numpy(copy=True)
            col_flags = flags[col].to_numpy(copy=True)
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                continue
            # Never fill deliberate exclusions, or rows a join added (no sample).
            excluded = (col_flags & int(Flag.EXCLUDED | Flag.NO_SAMPLE)) != 0
            valid_pos = np.flatnonzero(valid)
            # For each position: the last valid sample at or before it, and the
            # next valid sample at or after it.
            prev_idx = np.maximum.accumulate(np.where(valid, np.arange(len(arr)), -1))
            next_idx = np.minimum.accumulate(
                np.where(valid, np.arange(len(arr)), len(arr))[::-1]
            )[::-1]
            interior = (prev_idx >= 0) & (next_idx < len(arr))
            fill = ~valid & ~excluded & interior
            if not fill.any():
                continue
            span = np.full(len(arr), np.inf)
            span[interior] = t[next_idx[interior]] - t[prev_idx[interior]]
            fill &= span <= max_gap_s
            if not fill.any():
                continue
            interp = _interpolator(self.method, t[valid_pos], arr[valid_pos])
            arr[fill] = np.asarray(interp(t[fill]), dtype=float)
            col_flags[fill] |= FLAG_DTYPE(int(Flag.IMPUTED_SHORT))
            values[col] = arr
            flags[col] = col_flags
        return sf.evolve(values=values, flags=flags)
