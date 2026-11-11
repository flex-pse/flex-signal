"""Cleaning steps: find bad values, set them to NaN, and flag why.

A cleaner never deletes a row or changes a good value. It sets each value it
rejects to NaN and adds its flag (``OUT_OF_RANGE``, ``RATE_OF_CHANGE``,
``OUTLIER``, ``STUCK`` or ``MANUAL``). Values that are already missing are left
alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from flexsignal.core.conditions import Condition, evaluate_condition
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.spec.models import Duration, Quantity, to_timedelta
from flexsignal.steps._base import StreamStep, check_streams, remove_values, seconds

__all__ = ["Expression", "Hampel", "Range", "RateOfChange", "Stuck", "Tukey"]

HAMPEL_MAD_SCALE = 1.4826
"""Scales a median absolute deviation to a standard deviation for normal data."""


def _bound(value: float | Quantity | None, units: str | None, stream: str, field: str):
    if value is None or isinstance(value, float | int):
        return value
    if value.units != units:
        raise SignalConfigError(
            f"{field} is given in {value.units!r} but stream {stream!r} is in "
            f"{units!r}. Give the bound in the stream's units (unit conversion "
            "arrives with the [units] extra).",
            field=field,
        )
    return value.value


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
    """

    kind: Literal["rate_of_change"] = "rate_of_change"
    max_rate: float = Field(gt=0, description="Largest allowed change per `per`.")
    per: Duration = Field("1h", description="Time unit of max_rate, e.g. 1h or 1min.")

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values that change too fast (see the class docstring)."""
        check_streams(sf, self.streams)
        per_s = to_timedelta(self.per).total_seconds()
        t = seconds(sf.index)
        values = sf.values
        remove = {}
        for col in self.streams:
            arr = values[col].to_numpy()
            reject = np.zeros(len(arr), dtype=bool)
            last_v = last_t = None
            for i in np.flatnonzero(~np.isnan(arr)):
                if last_v is not None:
                    rate = abs(arr[i] - last_v) / ((t[i] - last_t) / per_s)
                    if rate > self.max_rate:
                        reject[i] = True
                        continue
                last_v, last_t = arr[i], t[i]
            remove[col] = reject
        return remove_values(
            sf, pd.DataFrame(remove, index=sf.index), Flag.RATE_OF_CHANGE
        )


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
    1.4826 × the window's median absolute deviation. See Pearson, R. K. (2002),
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
            scale = np.maximum(HAMPEL_MAD_SCALE * mad, self.min_scale)
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
    flag: Literal["MANUAL", "OUT_OF_RANGE", "RATE_OF_CHANGE", "OUTLIER", "STUCK"] = (
        Field("MANUAL", description="Which flag to set on rejected values.")
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Flag values where the condition holds."""
        check_streams(sf, self.streams)
        hit = evaluate_condition(sf.values, self.invalid_when, field="invalid_when")
        remove = pd.DataFrame({col: hit for col in self.streams})
        return remove_values(sf, remove, Flag[self.flag])
