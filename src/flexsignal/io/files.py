"""Read and write SignalFrames as Parquet and CSV.

**Parquet** is the lossless format. Each stream is a float64 column next to a
``<stream>__flags`` uint16 column. Units, metadata and provenance go in the
Arrow schema metadata under the ``flexsignal`` key. A Parquet file written by
any other tool also reads, as long as it has a datetime index or column.

**CSV** is for interchange. Timestamps are written in ISO 8601 with their UTC
offset. Floats round-trip exactly: pandas writes their shortest repr and, from
pandas 3, parses at full precision by default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flexsignal.core.exceptions import SignalDataError
from flexsignal.core.frame import FLAGS_SUFFIX, SignalFrame, _with_time_index

METADATA_KEY = b"flexsignal"
"""Arrow schema-metadata key holding flex-signal's stream metadata and provenance."""

FORMAT_VERSION = 1
"""Version of the Parquet metadata layout. Files from a newer version are refused."""

CSV_TIME_COLUMN = "timestamp"
"""Name of the time column :func:`write_csv` writes."""


def _split_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    flag_cols = [
        c for c in df.columns if isinstance(c, str) and c.endswith(FLAGS_SUFFIX)
    ]
    if not flag_cols:
        return df, None
    streams = [c[: -len(FLAGS_SUFFIX)] for c in flag_cols]
    missing = [s for s in streams if s not in df.columns]
    if missing:
        raise SignalDataError(f"flags columns without a matching stream: {missing}.")
    values = df.drop(columns=flag_cols)
    flags = df[flag_cols].set_axis(streams, axis=1)[list(values.columns)]
    return values, flags


def _to_signal_frame(
    df: pd.DataFrame,
    *,
    time_column: str | None,
    timezone: str | None,
    meta: Mapping[str, Any],
    history: Any = (),
    iso: bool = False,
) -> SignalFrame:
    """Build one SignalFrame from a flat table of values and ``__flags`` columns.

    The time index is set (and made UTC) first, then values and flags are split
    and passed to a single SignalFrame, which sorts them together.
    """
    df = _with_time_index(df, timezone=timezone, time_column=time_column, iso=iso)
    values, flags = _split_flags(df)
    return SignalFrame(values, flags, meta, history)


def _units_meta(units: Mapping[str, str] | None) -> dict[str, dict[str, Any]]:
    return {name: {"units": unit} for name, unit in (units or {}).items()}


def write_parquet(sf: SignalFrame, path: str | Path) -> Path:
    """Write a SignalFrame to Parquet, keeping flags, metadata and provenance.

    Args:
        sf: The frame to write.
        path: Output path. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(sf.to_pandas(include_flags=True), preserve_index=True)
    info = {
        "format_version": FORMAT_VERSION,
        "streams": {name: meta.model_dump() for name, meta in sf.meta.items()},
        "history": list(sf.history),
    }
    metadata = dict(table.schema.metadata or {})
    metadata[METADATA_KEY] = json.dumps(info).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


def read_parquet(
    path: str | Path,
    *,
    units: Mapping[str, str] | None = None,
    timezone: str | None = None,
    time_column: str | None = None,
) -> SignalFrame:
    """Read a SignalFrame from Parquet.

    Args:
        path: The file.
        units: Units per stream. These override units stored in the file.
        timezone: Time zone for a timezone-naive time index (files from other
            tools). If not given, naive timestamps are taken as this machine's
            local time, with a warning.
        time_column: Column holding the timestamps, if they aren't the index.

    Returns:
        The SignalFrame.

    Raises:
        SignalDataError: If the file has no usable time index, or was written by
            a newer flex-signal whose metadata layout this release can't read.
    """
    path = Path(path)
    if not path.exists():
        raise SignalDataError(f"file not found: {path}")
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    df = table.to_pandas()
    if raw is None:
        return _to_signal_frame(
            df, time_column=time_column, timezone=timezone, meta=_units_meta(units)
        )
    info: dict[str, Any] = json.loads(raw)
    version = info.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SignalDataError(
            f"{path.name} has flex-signal metadata with an unreadable format_version "
            f"({version!r}). Rewrite the file with write_parquet."
        )
    if version > FORMAT_VERSION:
        raise SignalDataError(
            f"{path.name} was written by a newer flex-signal (Parquet format_version "
            f"{version}); this release reads up to {FORMAT_VERSION}. Upgrade "
            "flex-signal to read it."
        )
    meta = info.get("streams", {})
    for name, unit in (units or {}).items():
        meta.setdefault(name, {})["units"] = unit
    return _to_signal_frame(
        df,
        time_column=time_column,
        timezone=timezone,
        meta=meta,
        history=info.get("history", ()),
    )


def write_csv(
    sf: SignalFrame, path: str | Path, *, include_flags: bool = False
) -> Path:
    """Write a SignalFrame's values (and optionally flags) to CSV.

    Args:
        sf: The frame to write.
        path: Output path. Parent directories are created.
        include_flags: Also write one ``<stream>__flags`` column per stream.

    Returns:
        The path written.

    Raises:
        SignalDataError: If a stream is named ``timestamp``, the name of the
            file's time column.
    """
    if CSV_TIME_COLUMN in sf.columns:
        raise SignalDataError(
            f"a stream is named {CSV_TIME_COLUMN!r}, the name of the CSV's time "
            f"column, so it would read back as '{CSV_TIME_COLUMN}.1'. Rename the "
            "stream before writing, or write Parquet, which keeps any name.",
            field=CSV_TIME_COLUMN,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = sf.to_pandas(include_flags=include_flags)
    df.index = df.index.map(lambda ts: ts.isoformat())
    df.to_csv(path, index_label=CSV_TIME_COLUMN, lineterminator="\n")
    return path


def read_csv(
    path: str | Path,
    *,
    time_column: str = CSV_TIME_COLUMN,
    timezone: str | None = None,
    units: Mapping[str, str] | None = None,
) -> SignalFrame:
    """Read a SignalFrame from CSV.

    Columns named ``<stream>__flags`` are read back as that stream's flags.

    Args:
        path: The file. A UTF-8 byte-order mark is accepted.
        time_column: Column holding the timestamps.
        timezone: Time zone of timestamps written without an offset. If not
            given, they are taken as this machine's local time, with a warning.
        units: Units per stream.

    Returns:
        The SignalFrame.

    Raises:
        SignalDataError: If the file or the time column is missing, the
            timestamps can't be parsed, or naive timestamps fall in a
            daylight-saving change. Timestamps with UTC offsets that change
            (e.g. ``-08:00`` then ``-07:00`` across DST) are fine.
    """
    path = Path(path)
    if not path.exists():
        raise SignalDataError(f"file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if time_column not in df.columns:
        raise SignalDataError(
            f"{path.name} has no {time_column!r} column; columns are "
            f"{list(df.columns)}. "
            "Pass time_column= to name the timestamp column."
        )
    return _to_signal_frame(
        df,
        time_column=time_column,
        timezone=timezone,
        meta=_units_meta(units),
        iso=True,
    )
