"""Cleaning steps: find bad values, set them to NaN, and flag why.

A cleaner never deletes a row or changes a good value. It sets each value it
rejects to NaN and adds its flag (``OUT_OF_RANGE``, ``RATE_OF_CHANGE``,
``OUTLIER``, ``STUCK`` or ``MANUAL``). Values that are already missing are left
alone.

Two steps here reshape a signal rather than reject values, and say so in the
flags: ``filter`` smooths noise (``FILTERED``), and ``detotalize`` turns a
running total into a rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from flexsignal.backends import get_backend
from flexsignal.core.conditions import Condition, evaluate_condition
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.core.units import convert
from flexsignal.spec.models import Duration, Quantity, to_timedelta
from flexsignal.steps._base import (
    StreamStep,
    check_streams,
    registered_lookup,
    regular_step,
    remove_values,
    seconds,
)

__all__ = [
    "Detotalize",
    "Expression",
    "Filter",
    "Hampel",
    "Range",
    "RateOfChange",
    "Stuck",
    "Tukey",
]

_MAD_TO_SIGMA = 1.4826
"""1 / Φ⁻¹(0.75): turns a median absolute deviation into a standard deviation for
normal data. A fixed conversion, not a setting; tune ``Hampel.n_sigma`` instead."""


def _bound(value: float | Quantity | None, units: str | None, stream: str, field: str):
    if value is None or isinstance(value, float | int):
        return value
    if value.units == units:
        return value.value
    if not units:
        raise SignalConfigError(
            f"{field} is given in {value.units!r} but stream {stream!r} has no "
            "units; give the bound as a plain number, or set the stream's units.",
            field=field,
        )
    try:
        return convert(value.value, value.units, units)
    except SignalConfigError as exc:
        raise SignalConfigError(
            f"{field}: {exc.message} (stream {stream!r} is in {units!r}).", field=field
        ) from None


@register_step
class Range(StreamStep):
    """Reject values outside fixed limits, e.g. a negative flow or an error code.

    Flags rejected values ``OUT_OF_RANGE``. The limits themselves are allowed.
    """

    kind: Literal["range"] = "range"
    min: float | Quantity | None = Field(
        None, description="Smallest allowed value, as a number or {value, units}."
    )
    max: float | Quantity | None = Field(
        None, description="Largest allowed value, as a number or {value, units}."
    )

    @model_validator(mode="after")
    def _has_a_limit(self) -> Range:
        if self.min is None and self.max is None:
            raise ValueError("give min, max or both")
        return self

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values outside [min, max]."""
        check_streams(sf, self.streams)
        values = sf.values
        remove = {}
        for col in self.streams:
            lo = _bound(self.min, sf.meta[col].units, col, "min")
            hi = _bound(self.max, sf.meta[col].units, col, "max")
            series = values[col]
            out = pd.Series(False, index=series.index)
            if lo is not None:
                out |= series < lo
            if hi is not None:
                out |= series > hi
            if lo is not None and hi is not None and lo > hi:
                raise SignalConfigError(
                    f"min ({lo}) is larger than max ({hi}).", field="min"
                )
            remove[col] = out
        return remove_values(sf, pd.DataFrame(remove), Flag.OUT_OF_RANGE)


@register_step
class RateOfChange(StreamStep):
    """Reject values that change faster than the process physically can.

    Each value is compared with the last *accepted* value. If
    ``|change| / elapsed time`` exceeds ``max_rate`` per ``per``, the value is
    flagged ``RATE_OF_CHANGE``. A spike is caught without flagging the point
    after it, because that point is compared with the last good value before the
    spike. A genuine step change is accepted once enough time has passed for the
    allowed rate to cover it.

    With ``reset_after``, a run of that many rejected values in a row that agree
    with each other (each within ``max_rate`` of the one before) is taken as a new
    level rather than a spike: the run is kept and becomes the reference. If the
    old reference was the stream's first value, nothing was accepted after it,
    so it was the outlier and is flagged instead.
    """

    kind: Literal["rate_of_change"] = "rate_of_change"
    max_rate: float = Field(gt=0, description="Largest allowed change per `per`.")
    per: Duration = Field("1h", description="Time unit of max_rate, e.g. 1h or 1min.")
    reset_after: int | None = Field(
        None,
        ge=2,
        description=(
            "After this many rejected values in a row that agree with each other, "
            "accept them as a new level. Default: never."
        ),
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values that change too fast (see the class docstring)."""
        check_streams(sf, self.streams)
        per_s = to_timedelta(self.per).total_seconds()
        t = seconds(sf.index)
        values = sf.values
        remove = {}
        for col in self.streams:
            arr = values[col].to_numpy()
            remove[col] = self._reject(arr, t, per_s)
        return remove_values(
            sf, pd.DataFrame(remove, index=sf.index), Flag.RATE_OF_CHANGE
        )

    def _reject(self, arr: np.ndarray, t: np.ndarray, per_s: float) -> np.ndarray:
        def too_fast(i: int, j: int) -> bool:
            return abs(arr[j] - arr[i]) / ((t[j] - t[i]) / per_s) > self.max_rate

        reject = np.zeros(len(arr), dtype=bool)
        valid = np.flatnonzero(~np.isnan(arr))
        if not len(valid):
            return reject
        # The first valid value is the first reference, even if it is bad; with
        # reset_after, a run that disagrees with it can replace it.
        anchor = first = valid[0]
        run: list[int] = []
        for i in valid[1:]:
            if not too_fast(anchor, i):
                anchor, run = i, []
                continue
            reject[i] = True
            if self.reset_after is None:
                continue
            run = run + [i] if run and not too_fast(run[-1], i) else [i]
            if len(run) >= self.reset_after:
                reject[run] = False
                if anchor == first:  # nothing was ever accepted after it
                    reject[first] = True
                anchor, run = i, []
        return reject


@register_step
class Stuck(StreamStep):
    """Reject flat-lined runs, the signature of a frozen sensor or a held last value.

    A run is consecutive valid samples whose values differ by at most
    ``tolerance``. Missing samples in between don't break a run. Every sample of
    a run lasting ``min_duration`` or longer is flagged ``STUCK``. Runs at a value
    listed in ``exclude_values`` are left alone; this is useful for a legitimate
    constant such as 0 when a pump is off.
    """

    kind: Literal["stuck"] = "stuck"
    min_duration: Duration = Field(
        description="Shortest flat run that counts as stuck, e.g. 1h."
    )
    tolerance: float = Field(
        0.0,
        ge=0,
        description="Largest change between samples that still counts as flat.",
    )
    exclude_values: tuple[float, ...] = Field(
        (), description="Values at which a flat run is legitimate, e.g. [0]."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag stuck runs (see the class docstring)."""
        check_streams(sf, self.streams)
        min_ns = to_timedelta(self.min_duration).value
        values = sf.values
        remove = {}
        for col in self.streams:
            series = values[col]
            valid = series.dropna()
            reject = pd.Series(False, index=series.index)
            if len(valid) >= 2:
                arr = valid.to_numpy()
                breaks = np.concatenate([[True], np.abs(np.diff(arr)) > self.tolerance])
                run_id = np.cumsum(breaks)
                times = valid.index.asi8
                frame = pd.DataFrame({"run": run_id, "t": times, "v": arr})
                runs = frame.groupby("run").agg(
                    start=("t", "min"),
                    end=("t", "max"),
                    n=("t", "size"),
                    v=("v", "first"),
                )
                long = (runs["n"] >= 2) & ((runs["end"] - runs["start"]) >= min_ns)
                for value in self.exclude_values:
                    long &= ~(np.abs(runs["v"] - value) <= self.tolerance)
                hit = np.isin(run_id, runs.index[long].to_numpy())
                reject.loc[valid.index[hit]] = True
            remove[col] = reject
        return remove_values(sf, pd.DataFrame(remove), Flag.STUCK)


@register_step
class Hampel(StreamStep):
    """Reject spikes with a Hampel filter: a rolling median plus a robust spread.

    Each value is compared with the median of a centred time window. It is
    flagged ``OUTLIER`` if it lies more than ``n_sigma`` robust standard deviations
    from that median. The robust standard deviation is
    1.4826 × the window's median absolute deviation; 1.4826 is the fixed factor
    that makes this match the standard deviation for normal data, so ``n_sigma``
    is the only threshold to tune. See Pearson, R. K. (2002),
    "Outliers in process modeling and identification", IEEE Trans. Control
    Systems Technology 10(1), 55-63, doi:10.1109/87.974338.

    The window should be several times wider than the spikes being removed, and
    narrow enough that the signal's own trend changes little across it.

    A spread estimated from a short window can come out much smaller than the
    real noise, and then ordinary noise gets flagged. This happens most where
    the signal is flat. Set ``min_scale`` to the sensor's noise standard
    deviation, or its resolution, to put a floor under the spread.
    """

    kind: Literal["hampel"] = "hampel"
    window: Duration = Field(description="Width of the centred window, e.g. 1h.")
    n_sigma: float = Field(
        3.0,
        gt=0,
        description="How many robust standard deviations count as an outlier.",
    )
    min_periods: int = Field(
        3, ge=3, description="Fewest valid samples a window needs before it can flag."
    )
    min_scale: float = Field(
        0.0,
        ge=0,
        description=(
            "Floor on the robust standard deviation, in the stream's units; typically "
            "the sensor's noise level."
        ),
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag spikes (see the class docstring)."""
        check_streams(sf, self.streams)
        window = to_timedelta(self.window)
        values = sf.values
        remove = {}
        for col in self.streams:
            series = values[col]
            rolling = series.rolling(window, center=True, min_periods=self.min_periods)
            median = rolling.median()
            mad = rolling.apply(
                lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True
            )
            scale = np.maximum(_MAD_TO_SIGMA * mad, self.min_scale)
            remove[col] = (series - median).abs() > self.n_sigma * scale
        return remove_values(sf, pd.DataFrame(remove), Flag.OUTLIER)


@register_step
class Tukey(StreamStep):
    """Reject values beyond Tukey's fences, learned from the training data.

    ``fit`` records each stream's quartiles Q1 and Q3. Values below
    ``Q1 - k·IQR`` or above ``Q3 + k·IQR`` are flagged ``OUTLIER``, where
    IQR = Q3 - Q1. Use k = 1.5 for "outliers" and k = 3 for "far out" values
    (Tukey, J. W. (1977), *Exploratory Data Analysis*, Addison-Wesley).

    This suits streams whose normal range doesn't drift. For a signal with a
    strong trend or regimes, prefer ``hampel``.
    """

    kind: Literal["tukey"] = "tukey"
    k: float = Field(1.5, gt=0, description="Fence width in multiples of the IQR.")

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Learn each stream's fences from the training data."""
        check_streams(sf, self.streams)
        fences = {}
        values = sf.values
        for col in self.streams:
            valid = values[col].dropna().to_numpy()
            if len(valid) < 4:
                raise SignalDataError(
                    f"tukey needs at least 4 valid training values in {col!r} to "
                    f"estimate quartiles; it has {len(valid)}.",
                    field="streams",
                )
            q1, q3 = np.quantile(valid, [0.25, 0.75])
            iqr = q3 - q1
            fences[col] = [float(q1 - self.k * iqr), float(q3 + self.k * iqr)]
        return {"fences": fences}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values outside the fitted fences."""
        check_streams(sf, self.streams)
        values = sf.values
        remove = {}
        for col in self.streams:
            lo, hi = state["fences"][col]
            remove[col] = (values[col] < lo) | (values[col] > hi)
        return remove_values(sf, pd.DataFrame(remove), Flag.OUTLIER)


@register_step
class Expression(StreamStep):
    """Reject values wherever a user-written condition holds.

    For example ``invalid_when="pressure < 0 or pump_on == 0"``. Flags the
    rejected values with ``flag``, ``MANUAL`` by default.
    """

    kind: Literal["expression"] = "expression"
    invalid_when: Condition = Field(
        description="Condition over stream names, e.g. 'pressure < 0'."
    )
    flag: Literal[
        "MANUAL", "OUT_OF_RANGE", "RATE_OF_CHANGE", "OUTLIER", "STUCK", "EXCLUDED"
    ] = Field("MANUAL", description="Which flag to set on rejected values.")

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values where the condition holds."""
        check_streams(sf, self.streams)
        hit = evaluate_condition(sf, self.invalid_when, field="invalid_when")
        remove = pd.DataFrame({col: hit for col in self.streams})
        return remove_values(sf, remove, Flag[self.flag])


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` of each run of True values."""
    edges = np.diff(np.concatenate([[0], valid.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts, stops, strict=True))


@register_step
class Filter(StreamStep):
    """Smooth noise out of a stream.

    - ``savgol``: a Savitzky-Golay filter (:func:`scipy.signal.savgol_filter`)
      fits a polynomial of degree ``polyorder`` across a moving ``window``. It
      keeps peaks better than a moving average.
    - ``butterworth``: a low-pass Butterworth filter of ``order``, run forwards
      and backwards so it adds no delay (:func:`scipy.signal.sosfiltfilt`). It
      removes variation faster than ``cutoff``, a period such as 30min.
    - ``median``: a centred moving median over ``window``. It removes spikes as
      well as noise, but flattens sharp peaks.

    Every smoothed value is flagged ``FILTERED``. Missing values stay missing.
    ``savgol`` and ``butterworth`` need a regular index and filter each unbroken
    run of values separately; a run too short for the filter is left as it is.
    """

    kind: Literal["filter"] = "filter"
    method: Literal["savgol", "butterworth", "median"] = Field(
        description="savgol, butterworth or median."
    )
    window: Duration | None = Field(
        None, description="savgol and median: width of the moving window, e.g. 30min."
    )
    polyorder: int = Field(
        2, ge=0, le=10, description="savgol: degree of the fitted polynomial."
    )
    cutoff: Duration | None = Field(
        None,
        description="butterworth: the shortest period of variation to keep, e.g. 1h.",
    )
    order: int = Field(4, ge=1, le=10, description="butterworth: filter order.")

    @model_validator(mode="after")
    def _options_for_method(self) -> Filter:
        if self.method in ("savgol", "median") and self.window is None:
            raise ValueError(f"{self.method} needs a window, e.g. window='30min'")
        if self.method == "butterworth" and self.cutoff is None:
            raise ValueError("butterworth needs a cutoff period, e.g. cutoff='1h'")
        return self

    def _smooth_run(self, run: np.ndarray, step: pd.Timedelta) -> np.ndarray | None:
        if self.method == "savgol":
            n = int(round(to_timedelta(self.window or "") / step))
            n = n if n % 2 else n + 1
            if n <= self.polyorder or len(run) < n:
                return None
            return get_backend("scipy:savgol_filter")(run, n, self.polyorder)
        cutoff = to_timedelta(self.cutoff or "")
        wn = 2.0 * step.total_seconds() / cutoff.total_seconds()
        if not 0 < wn < 1:
            raise SignalConfigError(
                f"cutoff {self.cutoff} must be longer than two samples ({2 * step}).",
                field="cutoff",
            )
        sos = get_backend("scipy:butter")(self.order, wn, output="sos")
        if len(run) <= 3 * (2 * len(sos) + 1):
            return None
        return get_backend("scipy:sosfiltfilt")(sos, run)

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Smooth each stream (see the class docstring)."""
        check_streams(sf, self.streams)
        values, flags = sf.values, sf.flags
        step = (
            None
            if self.method == "median"
            else regular_step(sf.index, kind=f"filter ({self.method})")
        )
        for col in self.streams:
            arr = values[col].to_numpy(copy=True)
            valid = ~np.isnan(arr)
            out = np.full(len(arr), np.nan)
            if self.method == "median":
                window = to_timedelta(self.window or "")
                series = values[col].dropna()
                med = series.rolling(window, center=True, min_periods=1).median()
                out[valid] = med.to_numpy()
            else:
                assert step is not None
                for start, stop in _runs(valid):
                    smoothed = self._smooth_run(arr[start:stop], step)
                    if smoothed is not None:
                        out[start:stop] = smoothed
            done = ~np.isnan(out) & valid
            arr[done] = out[done]
            col_flags = flags[col].to_numpy(copy=True)
            col_flags[done] |= FLAG_DTYPE(int(Flag.FILTERED))
            values[col] = arr
            flags[col] = col_flags
        return sf.evolve(values=values, flags=flags)


@register_step
class Detotalize(StreamStep):
    """Turn a running total (a totalizer or counter) into a rate.

    Each valid sample gets the increase since the previous valid sample, divided
    by the time between them, per ``per``: a total in m^3 with ``per="1h"`` gives
    m^3/hr. The rate is stamped at the end of its interval, and carries the flags
    of both samples. The first valid sample has no rate and is flagged
    ``MISSING``.

    A counter that wraps around at ``rollover`` (e.g. 999999) is unwrapped. A
    total that drops with no ``rollover`` set was reset; the rate across the
    reset is unknown, so it is left missing and flagged ``OUT_OF_RANGE``.
    """

    kind: Literal["detotalize"] = "detotalize"
    per: Duration = Field("1h", description="Time unit of the rate, e.g. 1h.")
    rollover: float | None = Field(
        None, gt=0, description="Value at which the counter wraps back to 0."
    )
    units: str | None = Field(
        None, description="Units of the rate, e.g. m^3/hr (default: unchanged)."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Convert totals to rates (see the class docstring)."""
        check_streams(sf, self.streams)
        per_s = to_timedelta(self.per).total_seconds()
        t = seconds(sf.index)
        values, flags = sf.values, sf.flags
        meta = dict(sf.meta)
        for col in self.streams:
            arr = values[col].to_numpy()
            col_flags = flags[col].to_numpy().astype("int64")
            pos = np.flatnonzero(~np.isnan(arr))
            rate = np.full(len(arr), np.nan)
            out_flags = col_flags.copy()
            if len(pos):
                out_flags[pos[0]] |= int(Flag.MISSING)
            if len(pos) > 1:
                rise = np.diff(arr[pos])
                if self.rollover is not None:
                    rise = np.where(rise < 0, rise + self.rollover, rise)
                reset = rise < 0
                later = pos[1:]
                with np.errstate(invalid="ignore"):
                    rate[later] = np.where(
                        reset, np.nan, rise / np.diff(t[pos]) * per_s
                    )
                out_flags[later] |= col_flags[pos[:-1]]
                out_flags[later[reset]] |= int(Flag.OUT_OF_RANGE)
            values[col] = rate
            flags[col] = out_flags.astype(FLAG_DTYPE)
            if self.units is not None:
                meta[col] = meta[col].model_copy(update={"units": self.units})
        return sf.evolve(values=values, flags=flags, meta=meta)


__getattr__ = registered_lookup(__name__, "cleaner")
