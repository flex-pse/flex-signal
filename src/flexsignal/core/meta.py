"""Per-stream metadata carried alongside the values."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StreamMeta(BaseModel):
    """What is known about one stream, independent of its values.

    Units are kept as metadata rather than as a pandas extension dtype. Models
    then run on plain float64, and units are converted only at the boundaries
    (PLAN.md §5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: str | None = Field(
        None,
        description=(
            "Units as a pyomo/pint-parseable string, e.g. m^3/hr. None when unknown."
        ),
    )
    quantity_kind: str | None = Field(
        None, description="Physical quantity, e.g. VolumeFlowRate or Pressure."
    )
    source: str | None = Field(
        None,
        description="Where the stream came from: a file path or an external reference.",
    )
    description: str | None = Field(None, description="Free-text description.")
