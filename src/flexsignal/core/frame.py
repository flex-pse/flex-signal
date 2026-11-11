"""SignalFrame: values, quality flags, stream metadata and provenance in one object.

A SignalFrame is a thin wrapper over plain pandas objects, not a new timeseries
library. Any library call can work on :attr:`SignalFrame.values` directly.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import tzlocal

from flexsignal.core.exceptions import NaiveTimestampWarning, SignalDataError
from flexsignal.core.flags import ALL_BITS, FLAG_DTYPE, Flag, flag_names
from flexsignal.core.meta import StreamMeta

INDEX_NAME = "time"
"""Name given to every SignalFrame's time index."""

FLAGS_SUFFIX = "__flags"
"""Suffix of the flags column for a stream when frames are flattened to one table."""


_PACKAGE_DIR = os.path.normcase(os.path.abspath(Path(__file__).parent.parent)) + os.sep


def local_timezone() -> str:
    """Return this machine's time zone as an IANA name, e.g. ``"America/Los_Angeles"``.

    Naive timestamps are taken to be in this zone. The ``TZ`` environment
    variable overrides it on every platform (``TZ=UTC`` makes results the same
    on every machine). Falls back to ``"UTC"`` if the zone can't be determined.
    """
    for name in (os.environ.get("TZ", "").strip(), _system_timezone()):
        if name:
            try:
                ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                continue
            return name
    return "UTC"


def _system_timezone() -> str | None:
    try:
        return tzlocal.get_localzone_name()
    except Exception:  # tzlocal raises several types on unusual systems
        return None


def _localize(
    index: pd.DatetimeIndex, timezone: str, *, assumed: bool
) -> pd.DatetimeIndex:
    """Attach ``timezone`` to a naive index, explaining daylight-saving failures."""
    try:
        return index.tz_localize(timezone)
    except Exception as exc:  # pytz/zoneinfo raise several types
        source = "this machine's time zone" if assumed else "time zone"
        raise SignalDataError(
            f"could not localize to {source} {timezone!r}: {exc}. Naive local "
            "timestamps can't represent daylight-saving changes; export the "
            "data with UTC offsets, or in UTC, if you can."
        ) from exc


def _warn_naive(timezone: str) -> None:
    message = (
        "the timestamps have no time zone, so they are taken as local time on "
        f"this machine ({timezone}). Pass timezone={timezone!r} (or the plant's "
        "time zone) to say so and silence this warning."
    )
    warnings.warn(message, NaiveTimestampWarning, stacklevel=_caller_stacklevel())


def _caller_stacklevel() -> int:
    """Return the ``stacklevel`` of the first frame outside flex-signal's library code.

    A warning raised deep inside flex-signal (a CSV read inside a pipeline run, say)
    then points at the user's line. flex-signal's own tests count as callers.
    """
    frame, level = sys._getframe(1), 1  # the function that will call warnings.warn
    while frame.f_back is not None and _is_library_file(frame.f_code.co_filename):
        frame, level = frame.f_back, level + 1
    return level


def _is_library_file(filename: str) -> bool:
    path = os.path.normcase(os.path.abspath(filename))
    return path.startswith(_PACKAGE_DIR) and f"{os.sep}tests{os.sep}" not in path


def _normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise SignalDataError(
            "the time index must be a pandas DatetimeIndex, got "
            f"{type(index).__name__}. "
            "Set a datetime column as the index, or pass time_column= to "
            "SignalFrame.from_frame."
        )
    if index.tz is None:
        timezone = local_timezone()
        _warn_naive(timezone)
        index = _localize(index, timezone, assumed=True)
    return index.tz_convert("UTC").as_unit("ns").rename(INDEX_NAME)


def _as_float_frame(values: pd.DataFrame) -> pd.DataFrame:
    columns = list(values.columns)
    for col in columns:
        if not isinstance(col, str):
            raise SignalDataError(
                f"stream names must be strings, got {col!r} ({type(col).__name__}). "
                "Rename the columns, e.g. df.columns = df.columns.astype(str)."
            )
    if len(set(columns)) != len(columns):
        dupes = sorted({c for c in columns if columns.count(c) > 1})
        raise SignalDataError(
            f"duplicate stream names {dupes}; stream names must be unique."
        )
    converted = {}
    for col in columns:
        series = values[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            converted[col] = series.astype("float64")
        else:
            raise SignalDataError(
                f"stream {col!r} is not numeric (dtype {series.dtype}). "
                "Convert it with "
                "pd.to_numeric, or drop it before building a SignalFrame.",
                field=col,
            )
    return pd.DataFrame(converted, index=values.index, columns=columns)


class SignalFrame:
    """Values, per-sample quality flags, stream metadata and provenance.

    A SignalFrame is immutable: every operation returns a new frame. The time
    index is always timezone-aware UTC with nanosecond resolution, sorted and
    unique. Values are float64; flags are a ``uint16`` bitmask of :class:`Flag`
    bits with the same shape as the values.

    Args:
        values: One float column per stream, indexed by a tz-aware DatetimeIndex.
        flags: Flags with the same index and columns. If omitted, every NaN value
            is flagged ``MISSING`` and every other value is raw.
        meta: Stream metadata keyed by stream name. Streams not listed get empty
            metadata.
        history: Provenance records, oldest first (normally written by a pipeline).

    Raises:
        SignalDataError: If the index is not a sorted-or-sortable, unique,
            tz-aware DatetimeIndex, a stream is not numeric, or ``flags`` and
            ``meta`` don't match the values.
    """

    __slots__ = ("_values", "_flags", "_meta", "_history")

    def __init__(
        self,
        values: pd.DataFrame,
        flags: pd.DataFrame | None = None,
        meta: Mapping[str, StreamMeta | Mapping[str, Any]] | None = None,
        history: Sequence[Mapping[str, Any]] = (),
    ):
        """Validate and store the parts (see the class docstring)."""
        if not isinstance(values, pd.DataFrame):
            raise SignalDataError(
                f"values must be a pandas DataFrame, got {type(values).__name__}."
            )
        index = _normalize_index(values.index)
        if index.has_duplicates:
            dupes = index[index.duplicated()].unique()[:3]
            raise SignalDataError(
                "the time index has duplicate timestamps (first: "
                f"{list(map(str, dupes))}). Drop or aggregate the duplicates "
                "first, e.g. with df.groupby(level=0).mean()."
            )
        values = _as_float_frame(values)
        values.index = index
        values.columns.name = None

        if flags is None:
            flag_values = np.where(values.isna().to_numpy(), int(Flag.MISSING), 0)
            flags = pd.DataFrame(
                flag_values.astype(FLAG_DTYPE), index=index, columns=values.columns
            )
        else:
            if not isinstance(flags, pd.DataFrame):
                raise SignalDataError("flags must be a pandas DataFrame.")
            if list(flags.columns) != list(values.columns):
                raise SignalDataError(
                    "flags must have exactly the same columns, in the same order, "
                    "as the values."
                )
            if len(flags) != len(values) or not _normalize_index(flags.index).equals(
                index
            ):
                raise SignalDataError(
                    "flags must have exactly the same index as the values."
                )
            flags = pd.DataFrame(
                flags.to_numpy().astype(FLAG_DTYPE), index=index, columns=values.columns
            )

        if not index.is_monotonic_increasing:
            order = np.argsort(index.asi8, kind="stable")
            values = values.iloc[order]
            flags = flags.iloc[order]

        meta = dict(meta or {})
        unknown = sorted(set(meta) - set(values.columns))
        if unknown:
            raise SignalDataError(
                f"metadata given for streams that don't exist: {unknown}.",
            )
        self._values = values
        self._flags = flags
        self._meta = {
            col: (
                meta[col]
                if isinstance(meta.get(col), StreamMeta)
                else StreamMeta(**(meta.get(col) or {}))
            )
            for col in values.columns
        }
        self._history = tuple(dict(record) for record in history)

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        units: Mapping[str, str] | None = None,
        meta: Mapping[str, StreamMeta | Mapping[str, Any]] | None = None,
        timezone: str | None = None,
        time_column: str | None = None,
    ) -> SignalFrame:
        """Build a SignalFrame from a plain DataFrame.

        Args:
            df: Values, one column per stream.
            units: Units per stream, a shortcut for ``meta={name: {"units": ...}}``.
            meta: Full stream metadata per stream.
            timezone: Time zone of a timezone-naive index (e.g. ``"UTC"`` or
                ``"America/Los_Angeles"``). If it is not given, naive timestamps
                are taken as this machine's local time (:func:`local_timezone`)
                with a :class:`NaiveTimestampWarning`. Ignored if the index is
                already tz-aware.
            time_column: Column to use as the time index instead of the index.

        Returns:
            A new SignalFrame. NaN values are flagged ``MISSING``.

        Raises:
            SignalDataError: If the data can't form a valid SignalFrame, or naive
                timestamps fall in a daylight-saving change.
        """
        if not isinstance(df, pd.DataFrame):
            raise SignalDataError(
                f"expected a pandas DataFrame, got {type(df).__name__}."
            )
        if time_column is not None:
            if time_column not in df.columns:
                raise SignalDataError(
                    f"time column {time_column!r} not found; columns are "
                    f"{list(df.columns)}."
                )
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[time_column]))).drop(
                columns=time_column
            )
        index = df.index
        if not isinstance(index, pd.DatetimeIndex):
            try:
                index = pd.DatetimeIndex(pd.to_datetime(index))
            except (TypeError, ValueError) as exc:
                raise SignalDataError(
                    "the index could not be parsed as timestamps. Pass time_column= to "
                    "name the column that holds them."
                ) from exc
        if index.tz is None and timezone is not None:
            index = _localize(index, timezone, assumed=False)
        df = df.set_axis(index, axis=0)

        merged: dict[str, dict[str, Any]] = {}
        for name, item in (meta or {}).items():
            merged[name] = (
                item.model_dump() if isinstance(item, StreamMeta) else dict(item)
            )
        for name, unit in (units or {}).items():
            merged.setdefault(name, {})["units"] = unit
        return cls(df, meta=merged)

    def evolve(
        self,
        *,
        values: pd.DataFrame | None = None,
        flags: pd.DataFrame | None = None,
        meta: Mapping[str, StreamMeta | Mapping[str, Any]] | None = None,
    ) -> SignalFrame:
        """Return a new frame with some parts replaced; history is kept.

        Metadata for streams that no longer exist is dropped, and new streams get
        empty metadata unless ``meta`` provides it.

        Args:
            values: New values (default: unchanged).
            flags: New flags (default: unchanged if ``values`` is not given;
                otherwise ``MISSING`` where the new values are NaN).
            meta: New metadata (default: the current metadata, filtered).

        Returns:
            The new SignalFrame.
        """
        new_values = self._values if values is None else values
        if flags is None and values is not None:
            new_flags = None
        else:
            new_flags = self._flags if flags is None else flags
        source_meta = self._meta if meta is None else meta
        kept_meta = {k: v for k, v in source_meta.items() if k in new_values.columns}
        return SignalFrame(new_values, new_flags, kept_meta, self._history)

    def with_record(self, record: Mapping[str, Any]) -> SignalFrame:
        """Return a copy of this frame with one provenance record appended.

        Args:
            record: A JSON-able description of what produced the new state.

        Returns:
            The new SignalFrame.
        """
        frame = object.__new__(SignalFrame)
        frame._values = self._values
        frame._flags = self._flags
        frame._meta = dict(self._meta)
        frame._history = (*self._history, dict(record))
        return frame

    # -- access ---------------------------------------------------------------

    @property
    def values(self) -> pd.DataFrame:
        """The values, one float64 column per stream (a copy-on-write copy)."""
        return self._values.copy(deep=False)

    @property
    def flags(self) -> pd.DataFrame:
        """The flags, one ``uint16`` column per stream (a copy-on-write copy)."""
        return self._flags.copy(deep=False)

    @property
    def meta(self) -> dict[str, StreamMeta]:
        """Stream metadata keyed by stream name."""
        return dict(self._meta)

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        """Provenance records, oldest first."""
        return tuple(dict(r) for r in self._history)

    @property
    def index(self) -> pd.DatetimeIndex:
        """The UTC time index."""
        return self._values.index

    @property
    def columns(self) -> list[str]:
        """Stream names, in order."""
        return list(self._values.columns)

    @property
    def units(self) -> dict[str, str | None]:
        """Units per stream (None when unknown)."""
        return {col: m.units for col, m in self._meta.items()}

    def __len__(self) -> int:
        """Return the number of timestamps."""
        return len(self._values)

    def __contains__(self, stream: object) -> bool:
        """Return whether ``stream`` is one of this frame's streams."""
        return stream in self._values.columns

    def __repr__(self) -> str:
        """Summarize the shape and time span."""
        if len(self):
            span = f"{self.index[0].isoformat()} → {self.index[-1].isoformat()}"
        else:
            span = "empty"
        return f"SignalFrame({len(self)} rows × {len(self.columns)} streams, {span})"

    def stream(self, name: str) -> pd.Series:
        """Return one stream's values as a Series.

        Raises:
            SignalDataError: If the stream doesn't exist.
        """
        if name not in self:
            raise SignalDataError(
                f"no stream named {name!r}; streams are {self.columns}.", field=name
            )
        return self._values[name].copy(deep=False)

    def select(self, streams: Sequence[str]) -> SignalFrame:
        """Return a frame with only the named streams, in the given order."""
        missing = [s for s in streams if s not in self]
        if missing:
            raise SignalDataError(
                f"no streams named {missing}; streams are {self.columns}."
            )
        streams = list(streams)
        return SignalFrame(
            self._values[streams],
            self._flags[streams],
            {s: self._meta[s] for s in streams},
            self._history,
        )

    def valid(self) -> pd.DataFrame:
        """Return a boolean frame that is True where a value is present (not NaN)."""
        return self._values.notna()

    def has_flag(self, flag: Flag) -> pd.DataFrame:
        """Return a boolean frame that is True where any bit of ``flag`` is set."""
        return (self._flags & int(flag)) != 0

    def to_pandas(self, *, include_flags: bool = False) -> pd.DataFrame:
        """Return the values as a plain DataFrame.

        Args:
            include_flags: Also add one ``<stream>__flags`` column per stream.

        Returns:
            A new DataFrame.
        """
        df = self._values.copy()
        if include_flags:
            for col in self.columns:
                df[f"{col}{FLAGS_SUFFIX}"] = self._flags[col]
        return df

    def flag_counts(self) -> dict[str, dict[str, Any]]:
        """Count values and flag bits per stream.

        Returns:
            ``{stream: {"rows": n, "valid": n_valid, "flags": {FLAG_NAME: count}}}``,
            JSON-able. Only flags that occur are listed.
        """
        counts: dict[str, dict[str, Any]] = {}
        flags = self._flags.to_numpy()
        for j, col in enumerate(self.columns):
            column = flags[:, j]
            per_flag = {
                bit.name: int(np.count_nonzero(column & bit.value))
                for bit in ALL_BITS
                if np.any(column & bit.value)
            }
            counts[col] = {
                "rows": len(self),
                "valid": int(self._values[col].notna().sum()),
                "flags": per_flag,
            }
        return counts

    def describe_flags(self, stream: str, timestamp: Any) -> list[str]:
        """Return the flag names set on one sample.

        Args:
            stream: Stream name.
            timestamp: Anything ``pandas.Timestamp`` accepts; naive means UTC.
        """
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        return flag_names(int(self._flags.at[ts, stream]))

    def equals(self, other: object) -> bool:
        """Return whether two frames have identical values, flags and metadata.

        NaNs compare equal to NaNs; history is not compared.
        """
        return (
            isinstance(other, SignalFrame)
            and self._values.equals(other._values)
            and self._flags.equals(other._flags)
            and self._meta == other._meta
        )

    # -- pickling (slots classes need explicit state) ---------------------------

    def __getstate__(self):
        """Return the picklable state."""
        return (self._values, self._flags, self._meta, self._history)

    def __setstate__(self, state):
        """Restore from :meth:`__getstate__`."""
        self._values, self._flags, self._meta, self._history = state
