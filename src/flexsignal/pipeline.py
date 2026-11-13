"""Pipeline and FittedPipeline: run an ordered list of steps and keep the fitted state.

A :class:`Pipeline` wraps a :class:`~flexsignal.spec.SignalSpec`. Fitting runs
the steps in order on training data. Each step's :meth:`fit` sees the data as the
earlier steps left it, and returns a JSON-able state. A :class:`FittedPipeline`
is the spec plus those states. It transforms new data, reports what it did, and
saves itself as plain JSON, so no pickle is involved.

Every step's effect is recorded in the output frame's ``history``. The pipeline
compares each step's input and output, so the record is exact, and steps don't
have to write it themselves.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.io import read_csv, read_parquet
from flexsignal.spec import (
    CsvInput,
    FrameInput,
    InputModel,
    OutputModel,
    ParquetInput,
    SignalSpec,
    StepSpec,
    dump_spec,
    load_spec,
    spec_hash,
)

PAYLOAD_FORMAT = "flexsignal.fitted"
"""Identifies a fitted-pipeline payload."""

PAYLOAD_VERSION = 1
"""Version of the payload layout."""

Progress = Callable[[float, str], None]
"""Progress callback: ``(fraction_done, message)``."""

FrameLike = SignalFrame | pd.DataFrame | tuple[pd.DataFrame, Mapping[str, str]]
"""Data for one input slot: a SignalFrame, a DataFrame, or ``(df, units)``."""

DataLike = FrameLike | Mapping[str, FrameLike] | None
"""Data for a pipeline: one frame, or a mapping of input slot name to frame."""


# -- binding inputs ----------------------------------------------------------------


def _to_signal_frame(item: Any, slot: str) -> SignalFrame:
    if isinstance(item, SignalFrame):
        return item
    if isinstance(item, pd.DataFrame):
        return SignalFrame.from_frame(item)
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], pd.DataFrame)
        and isinstance(item[1], Mapping)
    ):
        return SignalFrame.from_frame(item[0], units=item[1])
    raise SignalConfigError(
        f"input {slot!r}: expected a SignalFrame, a DataFrame or (DataFrame, units), "
        f"got {type(item).__name__}.",
        field=f"inputs.{slot}",
    )


def _read_input(spec_input: InputModel, base_dir: Path) -> SignalFrame:
    path = Path(spec_input.path)
    path = path if path.is_absolute() else base_dir / path
    if isinstance(spec_input, ParquetInput):
        return read_parquet(path)
    if isinstance(spec_input, CsvInput):
        return read_csv(
            path,
            time_column=spec_input.time_column,
            timezone=spec_input.timezone,
            units=spec_input.units or None,
        )
    raise AssertionError(spec_input)  # pragma: no cover


def _join(frames: Sequence[tuple[str, SignalFrame]]) -> SignalFrame:
    if len(frames) == 1:
        return frames[0][1]
    seen: dict[str, str] = {}
    for slot, frame in frames:
        for col in frame.columns:
            if col in seen:
                raise SignalDataError(
                    f"stream {col!r} appears in inputs {seen[col]!r} and {slot!r}; "
                    "stream names must be unique across inputs. Rename one first."
                )
            seen[col] = slot
    values = pd.concat([f.values for _, f in frames], axis=1, join="outer", sort=True)
    flags = pd.concat([f.flags for _, f in frames], axis=1, join="outer", sort=True)
    # Rows the join adds to a stream aren't samples of it; mark them so align,
    # resample and interpolate don't treat them as real missing values.
    flags = flags.fillna(int(Flag.MISSING | Flag.NO_SAMPLE)).astype(FLAG_DTYPE)
    meta = {k: v for _, f in frames for k, v in f.meta.items()}
    return SignalFrame(values, flags, meta)


def bind_inputs(
    spec: SignalSpec, data: DataLike = None, *, base_dir: str | Path | None = None
) -> SignalFrame:
    """Turn the caller's data and the spec's inputs into one SignalFrame.

    With no inputs in the spec, ``data`` must be a single frame. Otherwise each
    input slot takes the data bound to its name. File inputs (``parquet``, ``csv``)
    are read only when no data is bound to them. Several inputs are joined on time
    (an outer join). The rows the join adds to a stream are flagged ``MISSING``
    and ``NO_SAMPLE``; put an ``align`` step first to bring them onto one grid.

    Args:
        spec: The spec.
        data: One frame, or ``{slot_name: frame}``.
        base_dir: Directory that relative file paths resolve against (default:
            the working directory).

    Returns:
        The joined SignalFrame.

    Raises:
        SignalConfigError: If data is missing for a slot, or given for an unknown
            slot.
    """
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    if not spec.inputs:
        if data is None:
            raise SignalConfigError(
                "no data: the spec declares no inputs, so pass the data directly."
            )
        if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise SignalConfigError(
                    "the spec declares no inputs, so pass one frame, not "
                    f"{len(data)} named frames. Declare the slots under 'inputs'."
                )
            ((slot, item),) = data.items()
            return _to_signal_frame(item, slot)
        return _to_signal_frame(data, "data")

    names = [i.name for i in spec.inputs]
    if data is None:
        bound: Mapping[str, Any] = {}
    elif isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
        bound = data
    elif len(spec.inputs) == 1:
        bound = {names[0]: data}
    else:
        raise SignalConfigError(
            f"the spec has {len(names)} input slots {names}; pass data as "
            "{slot_name: frame}."
        )
    unknown = sorted(set(bound) - set(names))
    if unknown:
        raise SignalConfigError(
            f"data given for unknown input slot(s) {unknown}; the spec's slots "
            f"are {names}.",
            field="inputs",
        )
    frames = []
    for i, spec_input in enumerate(spec.inputs):
        if spec_input.name in bound:
            frames.append(
                (
                    spec_input.name,
                    _to_signal_frame(bound[spec_input.name], spec_input.name),
                )
            )
        elif isinstance(spec_input, FrameInput):
            raise SignalConfigError(
                f"no data bound to input slot {spec_input.name!r}; pass "
                f"data={{{spec_input.name!r}: frame}}.",
                field=f"inputs.{i}.name",
            )
        else:
            frames.append((spec_input.name, _read_input(spec_input, root)))
    return _join(frames)


# -- provenance ---------------------------------------------------------------------


def _changed_count(
    before: pd.Series, after: pd.Series, fb: pd.Series, fa: pd.Series
) -> int:
    vb, va = before.to_numpy(), after.to_numpy()
    same = (vb == va) | (np.isnan(vb) & np.isnan(va))
    return int(np.count_nonzero(~same | (fb.to_numpy() != fa.to_numpy())))


def step_record(
    index: int, step: StepSpec, before: SignalFrame, after: SignalFrame, digest: str
) -> dict[str, Any]:
    """Describe what one step did, by comparing its input and output.

    Returns:
        A JSON-able dict with the step's index, kind and name, the spec hash,
        row counts, streams added and removed, and (when the time index is
        unchanged) how many samples changed per stream.
    """
    record: dict[str, Any] = {
        "index": index,
        "kind": step.kind,
        "name": step.name,
        "spec_hash": digest,
        "rows_before": len(before),
        "rows_after": len(after),
        "streams_added": [c for c in after.columns if c not in before],
        "streams_removed": [c for c in before.columns if c not in after],
    }
    if before.index.equals(after.index):
        bv, av, bf, af = before.values, after.values, before.flags, after.flags
        changed = {}
        for col in after.columns:
            if col in before:
                n = _changed_count(bv[col], av[col], bf[col], af[col])
                if n:
                    changed[col] = n
        record["samples_changed"] = changed
    return record


def _check_json_state(state: Any, step: StepSpec, index: int) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise SignalConfigError(
            f"step {index} ({step.kind}) returned {type(state).__name__} from fit(); "
            "fit() must return a dict.",
            field=f"steps.{index}",
        )
    try:
        return json.loads(json.dumps(state))
    except (TypeError, ValueError) as exc:
        raise SignalConfigError(
            f"step {index} ({step.kind}) returned fitted state that isn't JSON-able: "
            f"{exc}. Convert numpy values with float()/int() or .tolist().",
            field=f"steps.{index}",
        ) from None


# -- pipeline ------------------------------------------------------------------------


class Pipeline:
    """An unfitted pipeline: a spec you can fit.

    Args:
        steps: The steps, in order.
        name: Pipeline name.
        inputs: Input slots (see :func:`~flexsignal.pipeline.bind_inputs`).
        outputs: Where :func:`flexsignal.run` writes results.
        description: What the pipeline is for.
        metadata: Free-form metadata kept with the spec.

    Examples:
        >>> import flexsignal as fs
        >>> pipe = fs.Pipeline([fs.clean.Range("flow", min=0)], name="demo")
        >>> pipe.spec.steps[0].kind
        'range'
    """

    def __init__(
        self,
        steps: Sequence[StepSpec | Mapping[str, Any]] = (),
        *,
        name: str = "pipeline",
        inputs: Sequence[InputModel | Mapping[str, Any]] = (),
        outputs: Sequence[OutputModel | Mapping[str, Any]] = (),
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Build the spec (see the class docstring)."""
        self.spec = SignalSpec(
            name=name,
            steps=list(steps),
            inputs=list(inputs),
            outputs=list(outputs),
            description=description,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_spec(cls, spec: SignalSpec | Mapping[str, Any] | str | Path) -> Pipeline:
        """Build a pipeline from a spec, a spec dict, JSON text, or a ``.json`` path."""
        pipeline = cls.__new__(cls)
        pipeline.spec = load_spec(spec)
        return pipeline

    def __repr__(self) -> str:
        """Show the name and step kinds."""
        kinds = ", ".join(s.kind for s in self.spec.steps)
        return f"Pipeline({self.spec.name!r}: {kinds})"

    def fit(
        self,
        data: DataLike = None,
        *,
        base_dir: str | Path | None = None,
        progress: Progress | None = None,
    ) -> FittedPipeline:
        """Fit every step, in order, on training data.

        Args:
            data: Training data (see :func:`~flexsignal.pipeline.bind_inputs`).
            base_dir: Directory for relative input file paths.
            progress: Called as ``progress(fraction, message)`` after each step.

        Returns:
            The fitted pipeline.
        """
        fitted, _ = self.fit_transform(data, base_dir=base_dir, progress=progress)
        return fitted

    def fit_transform(
        self,
        data: DataLike = None,
        *,
        base_dir: str | Path | None = None,
        progress: Progress | None = None,
    ) -> tuple[FittedPipeline, SignalFrame]:
        """Fit every step; return the fitted pipeline and transformed training data."""
        digest = spec_hash(self.spec)
        sf = bind_inputs(self.spec, data, base_dir=base_dir)
        input_summary = {"rows": len(sf), "streams": sf.columns}
        states = []
        records = []
        n = len(self.spec.steps)
        for i, step in enumerate(self.spec.steps):
            state = _check_json_state(step.fit(sf), step, i)
            after = step.transform(sf, state)
            record = step_record(i, step, sf, after, digest)
            sf = after.with_record(record)
            states.append(state)
            records.append(record)
            if progress is not None:
                progress((i + 1) / n, f"fit {step.kind}")
        fit_report = {"spec_hash": digest, "input": input_summary, "steps": records}
        return FittedPipeline(self.spec, states, fit_report), sf


class FittedPipeline:
    """A pipeline whose steps have been fitted: the spec plus each step's state.

    Args:
        spec: The spec it was fitted from.
        states: One JSON-able state per step, in order.
        fit_report: What happened during fitting (from :meth:`Pipeline.fit`).
    """

    def __init__(
        self,
        spec: SignalSpec,
        states: Sequence[Mapping[str, Any]],
        fit_report: Mapping[str, Any] | None = None,
    ):
        """Store the parts (see the class docstring)."""
        if len(states) != len(spec.steps):
            raise SignalConfigError(
                f"got {len(states)} fitted states for {len(spec.steps)} steps."
            )
        self.spec = spec
        self.states = tuple(dict(s) for s in states)
        self.spec_hash = spec_hash(spec)
        self.fit_report = dict(fit_report or {"spec_hash": self.spec_hash, "steps": []})

    def __repr__(self) -> str:
        """Show the name and step count."""
        return f"FittedPipeline({self.spec.name!r}, {len(self.spec.steps)} steps)"

    def transform(
        self,
        data: DataLike = None,
        *,
        base_dir: str | Path | None = None,
        progress: Progress | None = None,
    ) -> SignalFrame:
        """Apply the fitted steps to new data.

        Args:
            data: The data (see :func:`~flexsignal.pipeline.bind_inputs`).
            base_dir: Directory for relative input file paths.
            progress: Called as ``progress(fraction, message)`` after each step.

        Returns:
            The transformed SignalFrame, with one history record per step.
        """
        sf = bind_inputs(self.spec, data, base_dir=base_dir)
        n = len(self.spec.steps)
        for i, (step, state) in enumerate(
            zip(self.spec.steps, self.states, strict=True)
        ):
            after = step.transform(sf, state)
            sf = after.with_record(step_record(i, step, sf, after, self.spec_hash))
            if progress is not None:
                progress((i + 1) / n, f"transform {step.kind}")
        return sf

    def report(self, result: SignalFrame | None = None) -> dict[str, Any]:
        """Summarize the pipeline and, optionally, one result, as a JSON-able dict.

        Args:
            result: A frame this pipeline produced. Its flag counts and per-step
                history are included.
        """
        out: dict[str, Any] = {
            "pipeline": self.spec.name,
            "spec_hash": self.spec_hash,
            "flexsignal_version": _dist_version("flex-signal"),
            "fit": self.fit_report,
        }
        if result is not None:
            out["result"] = {
                "rows": len(result),
                "start": result.index[0].isoformat() if len(result) else None,
                "end": result.index[-1].isoformat() if len(result) else None,
                "streams": result.flag_counts(),
                "steps": list(result.history),
            }
        return json.loads(json.dumps(out))

    # -- persistence -----------------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """Return the fitted pipeline as a storage-agnostic payload.

        Returns:
            ``{"format", "format_version", "flexsignal_version", "spec",
            "spec_hash", "state": {"steps": [...]}, "fit_report", "blobs"}``.
            Everything except ``blobs`` is JSON-able. ``blobs`` maps a name to
            bytes, for steps whose state can't be JSON (none in M1).
        """
        return {
            "format": PAYLOAD_FORMAT,
            "format_version": PAYLOAD_VERSION,
            "flexsignal_version": _dist_version("flex-signal"),
            "spec": dump_spec(self.spec),
            "spec_hash": self.spec_hash,
            "state": {"steps": [dict(s) for s in self.states]},
            "fit_report": json.loads(json.dumps(self.fit_report)),
            "blobs": {},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FittedPipeline:
        """Rebuild a fitted pipeline from :meth:`to_payload` output.

        Raises:
            SignalConfigError: If the payload isn't a flex-signal fitted pipeline,
                or its spec doesn't match its recorded hash.
        """
        if payload.get("format") != PAYLOAD_FORMAT:
            raise SignalConfigError(
                f"not a flex-signal fitted pipeline (format {payload.get('format')!r})."
            )
        if payload.get("format_version") != PAYLOAD_VERSION:
            raise SignalConfigError(
                "unsupported fitted-pipeline format_version "
                f"{payload.get('format_version')!r}; "
                f"this release reads version {PAYLOAD_VERSION}."
            )
        spec = load_spec(payload["spec"])
        fitted = cls(spec, payload["state"]["steps"], payload.get("fit_report"))
        recorded = payload.get("spec_hash")
        if recorded and recorded != fitted.spec_hash:
            raise SignalConfigError(
                "the payload's spec doesn't match its recorded spec_hash, so the "
                "fitted "
                "state may not belong to it. Refit the pipeline."
            )
        return fitted

    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Save as a directory of JSON files.

        The layout is ``spec.json``, ``fitted/manifest.json``, one
        ``fitted/NN_<kind>.json`` state per step, and ``fitted/blobs/`` for binary
        state.

        Args:
            directory: Where to save. Created if needed.
            overwrite: Replace an existing saved pipeline in that directory.

        Returns:
            The directory.

        Raises:
            SignalConfigError: If a pipeline is already saved there and
                ``overwrite`` is False.
        """
        root = Path(directory)
        if (root / "spec.json").exists() and not overwrite:
            raise SignalConfigError(
                f"{root} already holds a saved pipeline; pass overwrite=True to "
                "replace it."
            )
        fitted_dir = root / "fitted"
        fitted_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        dump_spec(self.spec, root / "spec.json")
        entries = []
        for i, (step, state) in enumerate(
            zip(self.spec.steps, self.states, strict=True)
        ):
            filename = f"{i:02d}_{step.kind}.json"
            (fitted_dir / filename).write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8"
            )
            entries.append({"index": i, "kind": step.kind, "file": filename})
        manifest = {
            k: payload[k]
            for k in (
                "format",
                "format_version",
                "flexsignal_version",
                "spec_hash",
                "fit_report",
            )
        }
        manifest["steps"] = entries
        (fitted_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return root

    @classmethod
    def load(cls, directory: str | Path) -> FittedPipeline:
        """Load a pipeline saved with :meth:`save`."""
        root = Path(directory)
        manifest_path = root / "fitted" / "manifest.json"
        if not manifest_path.exists():
            raise SignalConfigError(
                f"{root} doesn't hold a saved pipeline (no {manifest_path})."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        states = [
            json.loads((root / "fitted" / entry["file"]).read_text(encoding="utf-8"))
            for entry in manifest["steps"]
        ]
        payload = {
            **manifest,
            "spec": json.loads((root / "spec.json").read_text(encoding="utf-8")),
            "state": {"steps": states},
            "blobs": {},
        }
        return cls.from_payload(payload)
