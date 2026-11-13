"""Pydantic models of a SignalSpec: inputs, steps, outputs.

The conventions mirror flex-pse's ``flexcore.config``. Models are strict
(``extra="forbid"``) and frozen. Each one is told apart by a ``kind`` string.
Every field has a ``description``, which goes verbatim into the committed JSON
Schema, so keep descriptions plain text.

A step *is* its spec. A step class subclasses :class:`StepSpec`, and the same
object is what you build in Python and what a JSON spec loads into. Fitted state
never lives on the step. :meth:`StepSpec.fit` returns it as a JSON-able dict, and
:meth:`StepSpec.transform` receives it back.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import get_step_class

SCHEMA_VERSION = "0.1.0"
"""Version of the SignalSpec schema this release reads and writes."""


class StrictModel(BaseModel):
    """Base for every spec model: unknown fields are errors; models are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_duration(value: str) -> str:
    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{value!r} is not a duration; use a pandas offset string such as "
            "'15min', '2h' or '1D'"
        ) from None
    if delta <= pd.Timedelta(0):
        raise ValueError(f"duration {value!r} must be positive")
    return value


def _check_signed_duration(value: str) -> str:
    try:
        pd.Timedelta(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{value!r} is not a duration; use a pandas offset string such as "
            "'15min' or '-5min'"
        ) from None
    return value


def _check_timestamp(value: str) -> str:
    try:
        pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{value!r} is not a timestamp; use ISO 8601, e.g. '2025-01-01T00:00:00Z'"
        ) from None
    return value


Duration = Annotated[str, AfterValidator(_check_duration)]
"""A positive duration as a pandas offset string, e.g. ``"15min"``."""

SignedDuration = Annotated[str, AfterValidator(_check_signed_duration)]
"""A duration that may be negative, e.g. ``"-5min"``."""

TimestampStr = Annotated[str, AfterValidator(_check_timestamp)]
"""An ISO 8601 timestamp. A timestamp without a time zone means UTC."""


def to_timedelta(value: str) -> pd.Timedelta:
    """Parse a validated duration string."""
    return pd.Timedelta(value)


def to_utc(value: str) -> pd.Timestamp:
    """Parse a validated timestamp string; naive timestamps are taken as UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


class Quantity(StrictModel):
    """A number with units, e.g. ``{"value": 5, "units": "m^3/hr"}``."""

    value: float = Field(description="The magnitude.")
    units: str = Field(
        description="Units as a pyomo/pint-parseable string, e.g. m^3/hr."
    )


class StepSpec(StrictModel):
    """Base class of every pipeline step.

    A subclass declares ``kind: Literal["my_kind"] = "my_kind"`` and its options
    as fields, then implements :meth:`transform`, and :meth:`fit` if it learns
    anything from training data.
    """

    kind: str = Field(description="Which step this is.")
    name: str | None = Field(
        None,
        description=(
            "Optional label for this step, shown in provenance and reports. Must be "
            "unique within a spec."
        ),
    )

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Learn whatever the step needs from training data.

        Args:
            sf: The training data, as it arrives at this step.

        Returns:
            The fitted state as a JSON-able dict. Stateless steps return ``{}``.
        """
        return {}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Apply the step.

        Args:
            sf: The data as it arrives at this step.
            state: What :meth:`fit` returned.

        Returns:
            A new SignalFrame.
        """
        raise NotImplementedError(f"step {self.kind!r} does not implement transform()")


# -- inputs and outputs ---------------------------------------------------------


class FrameInput(StrictModel):
    """A named input slot that the caller fills with data at run time."""

    kind: Literal["frame"] = "frame"
    name: str = Field(
        description="Slot name; the caller binds data to it, e.g. historian."
    )
    description: str | None = Field(None, description="What data belongs in this slot.")


class ParquetInput(StrictModel):
    """A Parquet file, read when the caller doesn't bind data to this slot."""

    kind: Literal["parquet"] = "parquet"
    name: str = Field(
        description="Slot name; data bound to this name replaces the file."
    )
    path: str = Field(description="Path to the file, relative to the spec file.")


class CsvInput(StrictModel):
    """A CSV file, read when the caller doesn't bind data to this slot."""

    kind: Literal["csv"] = "csv"
    name: str = Field(
        description="Slot name; data bound to this name replaces the file."
    )
    path: str = Field(description="Path to the file, relative to the spec file.")
    time_column: str = Field("timestamp", description="Column holding the timestamps.")
    timezone: str | None = Field(
        None,
        description=(
            "Time zone of timestamps written without an offset, e.g. UTC or "
            "America/Los_Angeles. If omitted, they are taken as the local time "
            "of the machine running the spec, with a warning; set it so the "
            "spec gives the same result everywhere."
        ),
    )
    units: dict[str, str] = Field(
        default_factory=dict, description="Units per stream, e.g. {flow: m^3/hr}."
    )


InputModel = FrameInput | ParquetInput | CsvInput
"""Any input model (for type hints)."""

InputSpec = Annotated[InputModel, Field(discriminator="kind")]
"""An input in a spec, told apart by ``kind``."""


class ParquetOutput(StrictModel):
    """Write the result to a Parquet file, keeping flags, units and provenance."""

    kind: Literal["parquet"] = "parquet"
    path: str = Field(description="Output path, relative to the spec file.")


class CsvOutput(StrictModel):
    """Write the result's values (and optionally flags) to a CSV file."""

    kind: Literal["csv"] = "csv"
    path: str = Field(description="Output path, relative to the spec file.")
    include_flags: bool = Field(
        False, description="Also write one <stream>__flags column per stream."
    )


OutputModel = ParquetOutput | CsvOutput
"""Any output model (for type hints)."""

OutputSpec = Annotated[OutputModel, Field(discriminator="kind")]
"""An output in a spec, told apart by ``kind``."""


# -- the spec ------------------------------------------------------------------


class SignalSpec(StrictModel):
    """A declarative, versioned description of a pipeline.

    ``steps`` accepts step objects or plain dicts with a ``kind``. Dicts are
    resolved through the step registry. :func:`flexsignal.spec.load_spec` is the
    preferred way to read a spec from JSON, because its errors point at the exact
    field.
    """

    schema_version: str = Field(
        SCHEMA_VERSION,
        pattern=r"^\d+\.\d+\.\d+$",
        description="Version of the SignalSpec schema, X.Y.Z.",
    )
    name: str = Field(description="Name of the pipeline.")
    description: str | None = Field(None, description="What the pipeline is for.")
    inputs: tuple[InputSpec, ...] = Field(
        (),
        description=(
            "Named input slots. With none, the pipeline takes a single frame of data."
        ),
    )
    steps: tuple[SerializeAsAny[StepSpec], ...] = Field(
        (), description="The steps, applied in order."
    )
    outputs: tuple[OutputSpec, ...] = Field(
        (), description="Where flexsignal.api.run writes the result."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata, kept with the spec."
    )

    @field_validator("steps", mode="before")
    @classmethod
    def _resolve_step_kinds(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, StepSpec | Mapping) or not hasattr(value, "__iter__"):
            raise ValueError("steps must be a list of steps")
        resolved = []
        for i, item in enumerate(value):
            if isinstance(item, StepSpec):
                resolved.append(item)
            elif isinstance(item, Mapping):
                kind = item.get("kind")
                if not isinstance(kind, str):
                    raise ValueError(f"step {i} needs a 'kind'")
                resolved.append(get_step_class(kind).model_validate(item))
            else:
                raise ValueError(
                    f"step {i} must be a step object or a dict, got "
                    f"{type(item).__name__}"
                )
        return tuple(resolved)

    @model_validator(mode="after")
    def _check_unique_names(self) -> SignalSpec:
        input_names = [i.name for i in self.inputs]
        dupes = sorted({n for n in input_names if input_names.count(n) > 1})
        if dupes:
            raise ValueError(f"input slot names must be unique; repeated: {dupes}")
        step_names = [s.name for s in self.steps if s.name is not None]
        dupes = sorted({n for n in step_names if step_names.count(n) > 1})
        if dupes:
            raise ValueError(f"step names must be unique; repeated: {dupes}")
        return self
