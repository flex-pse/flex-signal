"""The SignalSpec: a declarative, versioned JSON description of a pipeline.

This package mirrors flex-pse's ``flexcore.config`` conventions:

- strict pydantic models;
- a ``schema_version`` with migrations;
- ``kind``-discriminated steps;
- ``$source`` file references;
- a committed JSON Schema.
"""

from flexsignal.spec.io import (
    MIGRATIONS,
    canonical_json,
    dump_spec,
    load_spec,
    migrate,
    resolve_sources,
    spec_hash,
)
from flexsignal.spec.models import (
    SCHEMA_VERSION,
    CsvInput,
    CsvOutput,
    Duration,
    FrameInput,
    InputModel,
    InputSpec,
    OutputModel,
    OutputSpec,
    ParquetInput,
    ParquetOutput,
    Quantity,
    SignalSpec,
    SignedDuration,
    StepSpec,
    StrictModel,
    TimestampStr,
    to_timedelta,
    to_utc,
)

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "CsvInput",
    "CsvOutput",
    "Duration",
    "FrameInput",
    "InputModel",
    "InputSpec",
    "OutputModel",
    "OutputSpec",
    "ParquetInput",
    "ParquetOutput",
    "Quantity",
    "SignalSpec",
    "SignedDuration",
    "StepSpec",
    "StrictModel",
    "TimestampStr",
    "canonical_json",
    "dump_spec",
    "load_spec",
    "migrate",
    "resolve_sources",
    "spec_hash",
    "to_timedelta",
    "to_utc",
]
