"""Entry points that follow the orchestrator contract.

These are the functions flex-orchestrator will wrap:

- a spec goes in and a JSON-able result comes out;
- no I/O happens when data is passed in;
- progress and errors take shapes a command ledger can record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flexsignal.core.frame import SignalFrame
from flexsignal.io import write_csv, write_parquet
from flexsignal.pipeline import (
    DataLike,
    FittedPipeline,
    Pipeline,
    Progress,
    resolve_spec,
)
from flexsignal.spec import CsvOutput, ParquetOutput, SignalSpec

__all__ = ["RunResult", "fit", "run"]


@dataclass(frozen=True)
class RunResult:
    """What :func:`run` returns."""

    signal: SignalFrame
    """The result as a SignalFrame (values, flags, metadata, history)."""
    fitted: FittedPipeline
    """The pipeline fitted on the run's data. Save it to reuse it on new data."""
    outputs: tuple[str, ...] = ()
    """Paths written for the spec's outputs."""

    @property
    def data(self) -> pd.DataFrame:
        """The values, as a DataFrame with a UTC index and one column per stream."""
        return self.signal.values

    @property
    def flags(self) -> pd.DataFrame:
        """The quality flags, same shape as :attr:`data`."""
        return self.signal.flags

    @property
    def units(self) -> dict[str, str | None]:
        """Units per stream, as passed to ``orch.store_data(name, df, units=)``."""
        return self.signal.units

    @property
    def spec_hash(self) -> str:
        """sha256 of the spec that produced this result."""
        return self.fitted.spec_hash

    @property
    def provenance(self) -> list[dict[str, Any]]:
        """One JSON-able record per step applied."""
        return list(self.signal.history)

    @property
    def report(self) -> dict[str, Any]:
        """A JSON-able summary of the spec hash, fit report, flag counts and history."""
        return self.fitted.report(self.signal)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-able parts of the result (everything except the data)."""
        return {
            "spec_hash": self.spec_hash,
            "units": self.units,
            "outputs": list(self.outputs),
            "report": self.report,
        }


def fit(
    spec: SignalSpec | Mapping[str, Any] | str | Path,
    data: DataLike = None,
    *,
    base_dir: str | Path | None = None,
    progress: Progress | None = None,
    plugins: Sequence[str] = (),
) -> FittedPipeline:
    """Fit a pipeline from a spec.

    Args:
        spec: A SignalSpec, a spec dict, JSON text, or the path of a ``.json`` spec.
        data: Training data: one frame, or ``{slot_name: frame}``. File inputs
            not bound here are read from disk.
        base_dir: Directory for relative paths (default: the spec file's directory,
            or the working directory).
        progress: Called as ``progress(fraction, message)`` after each step.
        plugins: Modules to import first, so the methods they register are
            available, e.g. ``["myplant.methods"]`` (see
            :func:`flexsignal.load_plugins`). Chosen by the caller, never by the
            spec.

    Returns:
        The fitted pipeline.
    """
    resolved, root = resolve_spec(spec, base_dir, plugins)
    return Pipeline.from_spec(resolved).fit(data, base_dir=root, progress=progress)


def run(
    spec: SignalSpec | Mapping[str, Any] | str | Path,
    data: DataLike = None,
    *,
    base_dir: str | Path | None = None,
    write_outputs: bool = True,
    progress: Progress | None = None,
    plugins: Sequence[str] = (),
) -> RunResult:
    """Fit a pipeline on data and return the transformed result.

    Args:
        spec: A SignalSpec, a spec dict, JSON text, or the path of a ``.json`` spec.
        data: One frame, or ``{slot_name: frame}``. File inputs not bound here are
            read from disk.
        base_dir: Directory for relative paths (default: the spec file's directory,
            or the working directory).
        write_outputs: Write the spec's ``outputs``. Pass False to keep the run
            free of file I/O.
        progress: Called as ``progress(fraction, message)`` after each step.
        plugins: Modules to import first, so the methods they register are
            available (see :func:`flexsignal.load_plugins`). Chosen by the
            caller, never by the spec.

    Returns:
        A :class:`RunResult`.
    """
    resolved, root = resolve_spec(spec, base_dir, plugins)
    fitted, signal = Pipeline.from_spec(resolved).fit_transform(
        data, base_dir=root, progress=progress
    )
    written = []
    if write_outputs:
        for output in resolved.outputs:
            path = Path(output.path)
            path = path if path.is_absolute() else root / path
            if isinstance(output, ParquetOutput):
                write_parquet(signal, path)
            elif isinstance(output, CsvOutput):
                write_csv(signal, path, include_flags=output.include_flags)
            written.append(str(path))
    return RunResult(signal=signal, fitted=fitted, outputs=tuple(written))
