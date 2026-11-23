"""The options of :func:`~flexsignal.handoff.to_grid` and ``fit_data``.

They are pydantic models with flex-orchestrator's field names, defaults and
validation (its ``model_inputs`` alignment fields and its fit ``prep``), so a
study config and a script describe the same thing in the same words.
"""

from __future__ import annotations

import zoneinfo
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from flexsignal.core.times import check_timestamp
from flexsignal.spec.models import StrictModel

__all__ = ["FitPrep", "FitResample", "GridOptions", "GridResample"]

Agg = Literal["mean", "sum", "min", "max", "median", "first", "last"]
"""How the samples in one step or bin combine."""

GridGaps = Literal["interpolate", "ffill", "zero", "error"]
"""What :func:`~flexsignal.handoff.to_grid` does with a time step that has no value."""

Reduce = Literal["mean", "last", "first", "min", "max", "median"]
"""The statistic that turns a horizon's data into one number."""


class GridResample(StrictModel):
    """How the samples in one model time step combine.

    A time step at ``t`` with length ``dt`` takes the samples of one interval and
    combines them with ``agg``:

    - ``forward`` (the default): ``[t, t + dt)``, the interval starting at ``t``.
      This is what flex-pse means by a time step: its time points are interval
      starts, and it costs a rate as ``rate[t] * dt``. A sample stamped ``s``
      stands for the interval starting at ``s``.
    - ``backward``: ``(t - dt, t]``, the interval ending at ``t``, for a value that
      sums up what happened up to ``t``. A sample stamped ``s`` stands for the
      interval ending at ``s``.

    With ``agg="sum"`` and data coarser than the steps (hourly kWh on 15-minute
    steps), each sample's amount is split evenly over the steps its interval
    covers. That needs the data's spacing to be a whole number of steps.
    """

    agg: Agg = Field("mean", description="How the samples in one step combine.")
    window: Literal["forward", "backward"] = Field(
        "forward",
        description=(
            "forward: a step takes [t, t + dt); backward: (t - dt, t], the interval "
            "ending at it."
        ),
    )


class GridOptions(StrictModel):
    """How timeseries data becomes one value per model time step.

    Example:
        >>> from flexsignal.handoff import GridOptions
        >>> GridOptions(timezone="America/Los_Angeles", gaps="error").gaps
        'error'
    """

    timezone: str = Field(
        "UTC",
        description=(
            "The facility's time zone, e.g. America/Los_Angeles. The model runs on "
            "that site's wall clock, so UTC data is converted to it before it is "
            "lined up with the model's (naive) time steps."
        ),
    )
    resample: GridResample | None = Field(
        None, description="How the samples in one time step combine (default: mean)."
    )
    gaps: GridGaps = Field(
        "interpolate",
        description=(
            "A time step with no value is filled by interpolate (in time, inside the "
            "data only), ffill (the last value carried forward) or zero; error "
            "refuses any gap."
        ),
    )
    limit: int | None = Field(
        None,
        ge=1,
        description="The longest run of missing steps interpolate or ffill fills.",
    )
    reduce: Reduce | None = Field(
        None,
        description=(
            "Give one number instead of a series: this statistic of the data over the "
            "model's horizon."
        ),
    )
    offset: str | None = Field(
        None,
        description=(
            "Read the data this far from the model's horizon, as a pandas offset "
            "(-365D: last year's data for this horizon)."
        ),
    )
    data_years: dict[int, int] | None = Field(
        None,
        description=(
            "{modeled year: data year}: a horizon in a modeled year reads the same "
            "calendar dates in the data year, before offset."
        ),
    )

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown time zone {value!r}") from exc
        return value

    @field_validator("offset")
    @classmethod
    def _an_offset(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                pd.Timedelta(value)
            except ValueError as exc:
                raise ValueError(f"offset {value!r} isn't a duration") from exc
        return value

    @model_validator(mode="after")
    def _reduce_alone(self) -> GridOptions:
        if self.reduce is not None:
            given = {"resample", "gaps", "limit"} & self.model_fields_set
            if given:
                raise ValueError(
                    f"reduce gives one number, so {sorted(given)} don't apply"
                )
        return self


class FitResample(StrictModel):
    """Resample fit data to a regular index."""

    freq: str = Field(description="A pandas offset alias, e.g. 1h.")
    agg: Agg = Field("mean", description="How the values in one bin combine.")


class FitPrep(StrictModel):
    """How input and output data are prepared before a regressor is fitted."""

    align: Literal["inner", "outer"] = Field(
        "inner",
        description=(
            "inner keeps the timestamps every column has; outer keeps any column's, "
            "leaving gaps."
        ),
    )
    resample: FitResample | None = Field(
        None, description="Resample to a regular index first."
    )
    gaps: Literal["error", "drop", "interpolate", "longest_segment"] = Field(
        "error",
        description=(
            "What to do with missing values: error, drop the rows, interpolate (in "
            "time, up to limit missing values in a row), or keep the longest_segment "
            "with none."
        ),
    )
    limit: int | None = Field(
        None, ge=1, description="The longest run of missing values interpolate fills."
    )
    max_rows: int = Field(
        1_000_000,
        ge=1,
        description=(
            "Refuse more rows than this (a guard against running out of memory)."
        ),
    )
    start: str | None = Field(
        None,
        description=(
            "Keep rows from this time on (ISO 8601; without a zone it means UTC)."
        ),
    )
    end: str | None = Field(
        None,
        description="Keep rows before this time (ISO 8601; without a zone, UTC).",
    )
    rename: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Column -> the name the regressor or model expects, e.g. "
            "{'feed': 'flow_in_feed'}; applied last, units included."
        ),
    )
    timezone: str | None = Field(
        None,
        description=(
            "Give the result on this zone's wall clock, without a zone (naive), as "
            "to_grid and flex-pse models use it; e.g. America/Los_Angeles or UTC. "
            "Default: keep the UTC index."
        ),
    )

    @field_validator("start", "end")
    @classmethod
    def _a_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            check_timestamp(value)
        return value

    @field_validator("timezone")
    @classmethod
    def _a_zone(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                zoneinfo.ZoneInfo(value)
            except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"unknown time zone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _start_before_end(self) -> FitPrep:
        if self.start and self.end and _utc(self.start) >= _utc(self.end):
            raise ValueError(f"start {self.start} must be before end {self.end}")
        return self


def _utc(value: str) -> pd.Timestamp:
    """Return a timestamp string as UTC; one without a zone means UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
