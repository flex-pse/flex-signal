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
from flexsignal.core.times import check_duration, check_timestamp

SCHEMA_VERSION = "0.1.0"
"""Version of the SignalSpec schema this release reads and writes."""


class StrictModel(BaseModel):
    """Base for every spec model: unknown fields are errors; models are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_duration(value: str) -> str:
    check_duration(value)
    return value


def _check_signed_duration(value: str) -> str:
    check_duration(value, signed=True)
    return value


def _check_timestamp(value: str) -> str:
    check_timestamp(value)
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


_JSON_SCALARS = (str, int, float, bool, type(None))


def _find_non_json(value: Any, path: str = "") -> tuple[str, Any] | None:
    """Return ``(dotted path, value)`` of the first part of ``value`` JSON can't hold.

    Returns None when ``value`` is made only of strings, numbers, booleans, None,
    lists and dicts with string keys (subclasses of these, such as numpy's
    float64, are fine). Numpy integers, timestamps and other objects are not.
    """
    seen: set[int] = set()

    def walk(item: Any, where: str) -> tuple[str, Any] | None:
        if isinstance(item, _JSON_SCALARS):
            return None
        if isinstance(item, Mapping | list | tuple):
            if id(item) in seen:
                return where, item
            seen.add(id(item))
            if isinstance(item, Mapping):
                pairs: Any = item.items()
            else:
                pairs = enumerate(item)
            for key, sub in pairs:
                if isinstance(item, Mapping) and not isinstance(key, str):
                    return f"{where}.{key!r}".lstrip("."), key
                found = walk(sub, f"{where}.{key}".lstrip("."))
                if found is not None:
                    return found
            seen.discard(id(item))
            return None
        return where, item

    return walk(value, path)


def _not_a_step(value: Any) -> str:
    name = value.__name__ if isinstance(value, type) else type(value).__name__
    if name == "SignalSpec":
        return (
            "got a whole SignalSpec where a list of steps belongs. Build a pipeline "
            "from a spec with fs.Pipeline.from_spec(spec), or pass the spec itself "
            "to fs.fit or fs.run"
        )
    return (
        f"got a {name}, which is not a step. Steps are step objects such as "
        "fs.clean.Range(...) or dicts with a 'kind'"
    )


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
    requires: tuple[str, ...] = Field(
        (),
        description=(
            "Modules that register the methods this spec uses, e.g. myplant.methods. "
            "Only a hint for error messages: flex-signal never imports them. Load "
            "them with fs.load_plugins or plugins=."
        ),
    )
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

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, value: Any) -> Any:
        # A spec dict validated directly, e.g. embedded in a caller's pydantic
        # params model, is checked and migrated exactly as load_spec does it.
        if isinstance(value, Mapping) and "schema_version" in value:
            from flexsignal.spec.io import migrate

            return migrate(dict(value))
        return value

    @field_validator("steps", mode="before")
    @classmethod
    def _resolve_step_kinds(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, BaseModel) and not isinstance(value, StepSpec):
            raise ValueError(_not_a_step(value))
        if isinstance(value, StepSpec | Mapping) or not hasattr(value, "__iter__"):
            raise ValueError("steps must be a list of steps")
        items = list(value)
        if items and all(
            isinstance(item, tuple) and len(item) == 2 and item[0] in cls.model_fields
            for item in items
        ):
            # list(spec) gives the spec's (field, value) pairs, so a whole spec
            # passed where its steps belong arrives here like this.
            raise ValueError(_not_a_step(cls))
        resolved = []
        for i, item in enumerate(items):
            if isinstance(item, StepSpec):
                resolved.append(item)
            elif isinstance(item, BaseModel):
                raise ValueError(f"step {i}: {_not_a_step(item)}")
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

    @field_validator("metadata")
    @classmethod
    def _check_metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        problem = _find_non_json(value, "metadata")
        if problem is not None:
            where, bad = problem
            raise ValueError(
                f"metadata must be JSON-able (strings, numbers, booleans, null, lists "
                f"and objects with string keys), but {where} is "
                f"{bad!r} (type {type(bad).__name__}). Convert it first, e.g. str(...) "
                "for a timestamp or float(...)/int(...) for a numpy number"
            )
        return value

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
