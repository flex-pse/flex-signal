"""Selection steps: put streams on a common grid, choose time windows and rows, reshape.

None of these steps judges data quality. They move, aggregate, rename or
exclude values, and record what they did in the flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from flexsignal.core.conditions import Condition, evaluate_condition
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import ALL_BITS, FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.core.units import convert
from flexsignal.spec.models import (
    Duration,
    SignedDuration,
    StepSpec,
    TimestampStr,
    to_timedelta,
    to_utc,
)
from flexsignal.steps._base import (
    StreamStep,
    check_streams,
    one_is_a_list,
    regular_step,
    whole_steps,
)

__all__ = [
    "Align",
    "Combine",
    "ConvertUnits",
    "Mask",
    "Rename",
    "Resample",
    "Scale",
    "Shift",
    "Window",
]


def _grid(
    start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta
) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=step, name="time").as_unit("ns")


@register_step
class Align(StepSpec):
    """Put every stream on one regular time grid.

    Each grid point takes the stream's sample at exactly that time if there is
    one, even if that sample is missing. (Rows flagged ``NO_SAMPLE``, which a join
    of several inputs adds, aren't samples.) Otherwise it takes a value from nearby
    valid samples, chosen by ``method``, and is flagged ``RESAMPLED``. With no
    valid sample within ``tolerance``, the point is NaN and flagged ``MISSING``.
    The grid is anchored at multiples of ``freq`` from the Unix epoch, so the
    same data always gives the same grid.
    """

    kind: Literal["align"] = "align"
    freq: Duration = Field(description="Grid spacing, e.g. 1min.")
    method: Literal["nearest", "previous", "linear"] = Field(
        "nearest",
        description=(
            "nearest: the closest valid sample; previous: the last valid sample at or "
            "before the grid time; linear: time-weighted between the valid samples "
            "either side."
        ),
    )
    tolerance: Duration | None = Field(
        None,
        description=(
            "nearest/previous: how far away a sample may be (default: freq). linear: "
            "the largest allowed gap between the two samples either side (default: "
            "2 x freq)."
        ),
    )
    how: Literal["outer", "inner"] = Field(
        "outer",
        description=(
            "outer: the grid spans every stream's data; inner: only the time all "
            "streams cover."
        ),
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Reindex every stream onto the grid (see the class docstring)."""
        step = to_timedelta(self.freq)
        values, flags = sf.values, sf.flags
        spans = [
            (values[c].first_valid_index(), values[c].last_valid_index())
            for c in sf.columns
        ]
        spans = [s for s in spans if s[0] is not None]
        if not spans:
            raise SignalDataError("align: no stream has any valid value to align.")
        if self.how == "outer":
            start = min(s[0] for s in spans).floor(step)
            end = max(s[1] for s in spans).ceil(step)
        else:
            start = max(s[0] for s in spans).ceil(step)
            end = min(s[1] for s in spans).floor(step)
            if start > end:
                raise SignalDataError(
                    "align: the streams' valid data never overlaps, so how='inner' "
                    "leaves no grid. Use how='outer'."
                )
        grid = _grid(start, end, step)
        default_tol = step * 2 if self.method == "linear" else step
        tol = to_timedelta(self.tolerance) if self.tolerance else default_tol

        new_values: dict[str, np.ndarray] = {}
        new_flags: dict[str, np.ndarray] = {}
        target = pd.DataFrame({"time": grid})
        for col in sf.columns:
            series, col_flags = values[col], flags[col]
            out_v = np.full(len(grid), np.nan)
            out_f = np.full(len(grid), int(Flag.MISSING), dtype=FLAG_DTYPE)
            # Rows a join added (NO_SAMPLE) are not samples of this stream.
            own_times = series.index[(col_flags.to_numpy() & int(Flag.NO_SAMPLE)) == 0]
            exact = grid.isin(own_times)
            exact_times = grid[exact]
            out_v[exact] = series.loc[exact_times].to_numpy()
            out_f[exact] = col_flags.loc[exact_times].to_numpy()

            valid = series.notna()
            source = pd.DataFrame(
                {
                    "time": series.index[valid],
                    "value": series[valid].to_numpy(),
                    "flag": col_flags[valid].to_numpy().astype("int64"),
                }
            )
            fill = ~exact
            if len(source) and fill.any():
                if self.method in ("nearest", "previous"):
                    direction = "nearest" if self.method == "nearest" else "backward"
                    matched = pd.merge_asof(
                        target, source, on="time", direction=direction, tolerance=tol
                    )
                    got = matched["value"].notna().to_numpy() & fill
                    out_v[got] = matched["value"].to_numpy()[got]
                    out_f[got] = (
                        matched["flag"].to_numpy()[got].astype("int64")
                        | int(Flag.RESAMPLED)
                    ).astype(FLAG_DTYPE)
                else:
                    before = pd.merge_asof(
                        target,
                        source.assign(t0=source["time"]),
                        on="time",
                        direction="backward",
                    )
                    after = pd.merge_asof(
                        target,
                        source.assign(t1=source["time"]),
                        on="time",
                        direction="forward",
                    )
                    t0, t1 = before["t0"], after["t1"]
                    span = (t1 - t0).to_numpy()
                    ok = (
                        (t0.notna() & t1.notna()).to_numpy(copy=True)
                        & fill
                        & (span <= tol.to_timedelta64())
                    )
                    t0_ns = (
                        t0.dt.tz_convert(None)
                        .to_numpy("datetime64[ns]")
                        .astype("int64")[ok]
                    )
                    t1_ns = (
                        t1.dt.tz_convert(None)
                        .to_numpy("datetime64[ns]")
                        .astype("int64")[ok]
                    )
                    t_ns = grid.asi8[ok]
                    v0 = before["value"].to_numpy()[ok]
                    v1 = after["value"].to_numpy()[ok]
                    weight = np.where(
                        t1_ns > t0_ns,
                        (t_ns - t0_ns) / np.maximum(t1_ns - t0_ns, 1),
                        0.0,
                    )
                    out_v[ok] = v0 + (v1 - v0) * weight
                    out_f[ok] = (
                        before["flag"].to_numpy()[ok].astype("int64")
                        | after["flag"].to_numpy()[ok].astype("int64")
                        | int(Flag.RESAMPLED)
                    ).astype(FLAG_DTYPE)
            new_values[col] = out_v
            new_flags[col] = out_f
        return sf.evolve(
            values=pd.DataFrame(new_values, index=grid, columns=sf.columns),
            flags=pd.DataFrame(new_flags, index=grid, columns=sf.columns),
        )


@register_step
class Resample(StepSpec):
    """Aggregate every stream into fixed time bins.

    Bins are left-closed and labelled by their start, and anchored at multiples of
    ``freq`` from the Unix epoch. A bin's flags are the OR of its samples' flags,
    plus ``RESAMPLED`` when it has a value. A bin with no samples at all is
    flagged ``MISSING``.
    """

    kind: Literal["resample"] = "resample"
    freq: Duration = Field(description="Bin width, e.g. 15min or 1h.")
    agg: Literal["mean", "median", "min", "max", "sum", "first", "last"] = Field(
        "mean", description="How to aggregate the valid samples in a bin."
    )
    min_count: int = Field(
        1, ge=1, description="Fewest valid samples a bin needs to get a value."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Aggregate into bins (see the class docstring)."""
        step = to_timedelta(self.freq)
        values, flags = sf.values, sf.flags

        def resampler(obj):
            return obj.resample(step, origin="epoch", label="left", closed="left")

        method = getattr(resampler(values), self.agg)
        agg = method(min_count=1) if self.agg == "sum" else method()
        counts = resampler(values).count()
        # Rows a join added (NO_SAMPLE) are not samples: they neither count
        # towards a bin nor contribute flags.
        own = (flags & int(Flag.NO_SAMPLE)) == 0
        samples = resampler(own).sum().reindex(agg.index, fill_value=0)
        agg = agg.where(counts >= self.min_count)

        combined = np.zeros(agg.shape, dtype="int64")
        for bit in ALL_BITS:
            has_bit = resampler(((flags & bit.value) != 0) & own).max()
            combined |= np.where(
                has_bit.fillna(False).to_numpy(dtype=bool), bit.value, 0
            )
        has_value = agg.notna().to_numpy()
        combined = np.where(has_value, combined | int(Flag.RESAMPLED), combined)
        empty = samples.to_numpy() == 0
        combined[empty] = int(Flag.MISSING)
        combined = np.where(~has_value & (combined == 0), int(Flag.MISSING), combined)
        index = agg.index.rename("time")
        return sf.evolve(
            values=agg.set_axis(index),
            flags=pd.DataFrame(
                combined.astype(FLAG_DTYPE), index=index, columns=sf.columns
            ),
        )


@register_step
class Window(StepSpec):
    """Keep only the rows inside a time window."""

    kind: Literal["window"] = "window"
    start: TimestampStr | None = Field(
        None, description="First time kept (inclusive). Naive means UTC."
    )
    end: TimestampStr | None = Field(
        None, description="End of the window (exclusive). Naive means UTC."
    )
    last: Duration | None = Field(
        None,
        description=(
            "Keep only this much time before the last timestamp, e.g. 7D. Can't be "
            "combined with start or end."
        ),
    )

    @model_validator(mode="after")
    def _one_way(self) -> Window:
        if self.last is not None and (self.start is not None or self.end is not None):
            raise ValueError("give either last, or start/end, not both")
        if self.last is None and self.start is None and self.end is None:
            raise ValueError("give start, end or last")
        return self

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Slice the rows (see the class docstring)."""
        index = sf.index
        keep = np.ones(len(index), dtype=bool)
        if self.last is not None and len(index):
            keep &= index > index[-1] - to_timedelta(self.last)
        if self.start is not None:
            keep &= index >= to_utc(self.start)
        if self.end is not None:
            keep &= index < to_utc(self.end)
        return sf.evolve(values=sf.values[keep], flags=sf.flags[keep])


@register_step
class Mask(StepSpec):
    """Keep data only where a condition holds; exclude the rest.

    Use this to select operating periods, e.g. ``keep_when="pump_on == 1"``.
    Excluded values become NaN and are flagged ``EXCLUDED``, which marks them as
    deliberately left out, not as bad data. Imputers never fill them. With
    ``drop_rows``, the rows are removed instead.
    """

    kind: Literal["mask"] = "mask"
    keep_when: Condition = Field(
        description="Condition over stream names, e.g. 'pump_on == 1 and flow > 0'."
    )
    streams: tuple[str, ...] | None = Field(
        None, description="Streams to exclude values from (default: all)."
    )
    drop_rows: bool = Field(
        False,
        description=(
            "Remove rows where the condition fails, instead of excluding values."
        ),
    )

    _one_stream_is_a_list = field_validator("streams", mode="before")(one_is_a_list)

    @model_validator(mode="after")
    def _drop_rows_is_whole_rows(self) -> Mask:
        if self.drop_rows and self.streams is not None:
            raise ValueError("drop_rows removes whole rows, so it can't take streams")
        return self

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Exclude values or drop rows (see the class docstring)."""
        keep = evaluate_condition(sf, self.keep_when, field="keep_when")
        if self.drop_rows:
            mask = keep.to_numpy()
            return sf.evolve(values=sf.values[mask], flags=sf.flags[mask])
        streams = list(self.streams) if self.streams else sf.columns
        check_streams(sf, streams)
        values, flags = sf.values, sf.flags
        exclude = ~keep.to_numpy()
        for col in streams:
            col_values = values[col].to_numpy(copy=True)
            col_values[exclude] = np.nan
            values[col] = col_values
            col_flags = flags[col].to_numpy(copy=True)
            col_flags[exclude] |= FLAG_DTYPE(int(Flag.EXCLUDED))
            flags[col] = col_flags
        return sf.evolve(values=values, flags=flags)


@register_step
class Rename(StepSpec):
    """Rename streams. Metadata and flags move with them."""

    kind: Literal["rename"] = "rename"
    mapping: dict[str, str] = Field(
        min_length=1, description="Old name -> new name, e.g. {FT_101: flow}."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Rename (see the class docstring)."""
        check_streams(sf, list(self.mapping), field="mapping")
        new_names = [self.mapping.get(c, c) for c in sf.columns]
        if len(set(new_names)) != len(new_names):
            raise SignalConfigError(
                f"renaming gives duplicate stream names: {new_names}.", field="mapping"
            )
        meta = {self.mapping.get(k, k): v for k, v in sf.meta.items()}
        return sf.evolve(
            values=sf.values.set_axis(new_names, axis=1),
            flags=sf.flags.set_axis(new_names, axis=1),
            meta=meta,
        )


@register_step
class ConvertUnits(StreamStep):
    """Convert streams to other units, e.g. a meter's gpm to the model's m^3/hr.

    Values are rescaled exactly (offset units such as degC and K too), the
    stream's units are updated, and the flags are unchanged. Needs pint (the
    ``[units]`` extra).
    """

    kind: Literal["convert_units"] = "convert_units"
    to: str = Field(description="The units wanted, e.g. m^3/hr.")

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Convert (see the class docstring)."""
        check_streams(sf, self.streams)
        values = sf.values
        meta = dict(sf.meta)
        for col in self.streams:
            units = meta[col].units
            if not units:
                raise SignalConfigError(
                    f"stream {col!r} has no units to convert from; set its units "
                    "when the data is read.",
                    field="streams",
                    value=col,
                )
            if units == self.to:
                continue
            values[col] = convert(values[col].to_numpy(), units, self.to)
            meta[col] = meta[col].model_copy(update={"units": self.to})
        # Converted values keep their flags (without flags=, evolve would rebuild
        # them from the NaNs and drop every flag set before this step).
        return sf.evolve(values=values, flags=sf.flags, meta=meta)


@register_step
class Scale(StreamStep):
    """Multiply streams by a constant and give them new units.

    For a change that isn't a unit conversion, which :class:`ConvertUnits`
    can't do: through a density (kg/h of sludge to m^3/h, ``factor=0.001`` at
    1000 kg/m^3), a heating value, or a meter's known scale factor. The flags
    are unchanged.
    """

    kind: Literal["scale"] = "scale"
    factor: float = Field(description="What every value is multiplied by.")
    units: str | None = Field(
        description="The streams' units afterwards, e.g. m^3/hr; null if unitless."
    )

    @field_validator("factor")
    @classmethod
    def _finite_and_nonzero(cls, value: float) -> float:
        if not np.isfinite(value) or value == 0:
            raise ValueError(f"factor must be a finite, non-zero number, got {value}")
        return value

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Scale (see the class docstring)."""
        check_streams(sf, self.streams)
        values = sf.values
        meta = dict(sf.meta)
        for col in self.streams:
            values[col] = values[col] * self.factor
            meta[col] = meta[col].model_copy(update={"units": self.units})
        return sf.evolve(values=values, flags=sf.flags, meta=meta)


@register_step
class Combine(StepSpec):
    """Build a new stream from several, e.g. two redundant meters.

    The result is flagged ``DERIVED``. For ``coalesce``, each sample takes the
    first stream (in order) that has a value, along with that stream's flags. For
    the other methods, the result's flags are the OR of every input's flags,
    without ``MISSING`` where the result has a value.

    ``sum`` is missing wherever any input is missing, since a total with a term
    left out is too low. ``missing_as_zero=True`` counts a missing input as 0
    instead; a time where every input is missing stays missing. ``mean``, ``min``
    and ``max`` use whichever inputs have a value, as for redundant meters.
    """

    kind: Literal["combine"] = "combine"
    output: str = Field(description="Name of the new stream.")
    streams: tuple[str, ...] = Field(
        min_length=2, description="The streams to combine."
    )
    how: Literal["coalesce", "mean", "sum", "min", "max"] = Field(
        "coalesce", description="How to combine the values at each time."
    )
    units: str | None = Field(
        None,
        description=(
            "Units of the result. Required if the inputs' units differ; defaults to "
            "their common units."
        ),
    )
    missing_as_zero: bool = Field(
        False,
        description=(
            "For how='sum' only: count a missing input as 0 instead of making the "
            "sum missing."
        ),
    )

    @model_validator(mode="after")
    def _zero_only_for_sum(self) -> Combine:
        if self.missing_as_zero and self.how != "sum":
            raise ValueError(
                f"missing_as_zero applies only to how='sum', not how={self.how!r}"
            )
        return self

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Combine (see the class docstring)."""
        streams = list(self.streams)
        check_streams(sf, streams)
        if self.output in sf:
            raise SignalConfigError(
                f"stream {self.output!r} already exists; pick another output name.",
                field="output",
            )
        input_units = {sf.meta[s].units for s in streams}
        if self.units is not None:
            units = self.units
        elif len(input_units) == 1:
            units = input_units.pop()
        else:
            raise SignalConfigError(
                f"the streams have different units {sorted(map(str, input_units))}; "
                "convert them first, or set units to assert the result's units.",
                field="units",
            )
        block = sf.values[streams]
        block_flags = sf.flags[streams].to_numpy().astype("int64")
        if self.how == "coalesce":
            arr = block.to_numpy()
            has = ~np.isnan(arr)
            first = np.where(has.any(axis=1), has.argmax(axis=1), -1)
            rows = np.arange(len(arr))
            result = np.where(first >= 0, arr[rows, np.maximum(first, 0)], np.nan)
            chosen = block_flags[rows, np.maximum(first, 0)]
            any_flags = np.bitwise_or.reduce(block_flags, axis=1)
            out_flags = np.where(first >= 0, chosen, any_flags)
        else:
            result = getattr(block, self.how)(axis=1, skipna=True).to_numpy()
            if self.how == "sum":
                present = block.notna()
                enough = (
                    present.any(axis=1) if self.missing_as_zero else present.all(axis=1)
                )
                result = np.where(enough.to_numpy(), result, np.nan)
            out_flags = np.bitwise_or.reduce(block_flags, axis=1)
            out_flags = np.where(
                np.isnan(result), out_flags, out_flags & ~int(Flag.MISSING)
            )
        out_flags = out_flags | int(Flag.DERIVED)
        out_flags = np.where(np.isnan(result), out_flags | int(Flag.MISSING), out_flags)
        values, flags = sf.values, sf.flags
        values[self.output] = result
        flags[self.output] = out_flags.astype(FLAG_DTYPE)
        meta = {**sf.meta, self.output: {"units": units}}
        return sf.evolve(values=values, flags=flags, meta=meta)


@register_step
class Shift(StreamStep):
    """Shift streams in time, e.g. to undo a known transport delay.

    A positive ``by`` moves values later. It needs a regular index, and ``by``
    must be a whole number of steps. Grid points left without a value are
    flagged ``MISSING``.
    """

    kind: Literal["shift"] = "shift"
    by: SignedDuration = Field(
        description="How far to move the values, e.g. 5min or -5min."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Shift (see the class docstring)."""
        check_streams(sf, self.streams)
        step = regular_step(sf.index, kind="shift")
        periods = whole_steps(
            to_timedelta(self.by), step, what=f"by={self.by}", field="by"
        )
        values, flags = sf.values, sf.flags
        for col in self.streams:
            values[col] = values[col].shift(periods)
            shifted = flags[col].shift(periods, fill_value=int(Flag.MISSING))
            flags[col] = shifted.astype(FLAG_DTYPE)
        return sf.evolve(values=values, flags=flags)
