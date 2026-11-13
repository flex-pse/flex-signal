"""Load, dump, migrate and hash SignalSpecs.

JSON is the only on-disk format, as in flex-pse. :func:`load_spec` resolves
``{"$source": "file.json"}`` references, migrates older schema versions, and
validates each step separately, so an error names the exact field (for example
``steps.2.window``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from flexsignal.core.exceptions import SignalConfigError
from flexsignal.core.registry import get_step_class
from flexsignal.spec.models import SCHEMA_VERSION, SignalSpec

MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""Schema migrations, keyed by the version they upgrade from.

Each entry is ``from_version -> fn(spec_dict) -> spec_dict``, returning the dict
at the next version. Add one whenever a released schema version changes
incompatibly."""

SOURCE_KEY = "$source"
"""Key of a reference to a JSON file holding a large value, e.g. a lookup table."""

SpecLike = SignalSpec | Mapping[str, Any] | str | Path


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(p) for p in version.split("."))
    except (AttributeError, ValueError):
        parts = ()
    if len(parts) != 3:
        raise SignalConfigError(
            f"schema_version must look like X.Y.Z, got {version!r}.",
            field="schema_version",
            value=version,
        )
    return parts


def _read_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SignalConfigError(f"file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SignalConfigError(f"{path} is not valid JSON: {exc}") from None


def resolve_sources(value: Any, base_dir: Path) -> Any:
    """Replace every ``{"$source": "file.json[.gz]"}`` with that file's contents.

    Args:
        value: A JSON-like structure.
        base_dir: Directory that relative ``$source`` paths resolve against.

    Returns:
        The structure with every reference replaced. Referenced files may contain
        references of their own, which resolve relative to that file.
    """
    if isinstance(value, Mapping):
        if set(value) == {SOURCE_KEY}:
            path = Path(value[SOURCE_KEY])
            path = path if path.is_absolute() else base_dir / path
            return resolve_sources(_read_json(path), path.parent)
        return {k: resolve_sources(v, base_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_sources(v, base_dir) for v in value]
    return value


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a spec dict up to :data:`~flexsignal.spec.models.SCHEMA_VERSION`.

    Raises:
        SignalConfigError: If the version is missing, newer than this release, or
            has no migration path.
    """
    version = data.get("schema_version")
    if version is None:
        raise SignalConfigError(
            "the spec has no schema_version. Add "
            f'"schema_version": "{SCHEMA_VERSION}".',
            field="schema_version",
        )
    current = _version_tuple(SCHEMA_VERSION)
    if _version_tuple(version) > current:
        raise SignalConfigError(
            f"the spec uses schema {version}, newer than this flex-signal release "
            f"supports ({SCHEMA_VERSION}). Upgrade flex-signal.",
            field="schema_version",
            value=version,
        )
    while data["schema_version"] != SCHEMA_VERSION:
        step = MIGRATIONS.get(data["schema_version"])
        if step is None:
            raise SignalConfigError(
                f"there is no migration from schema {data['schema_version']} to "
                f"{SCHEMA_VERSION}. Rewrite the spec for schema {SCHEMA_VERSION}.",
                field="schema_version",
                value=data["schema_version"],
            )
        data = step(data)
    return data


def _config_error(exc: ValidationError, prefix: str = "") -> SignalConfigError:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in (prefix, *err["loc"]) if p != "")
        # pydantic prefixes messages from our own validators with "Value error, ".
        problems.append((loc, err["msg"].removeprefix("Value error, ")))
    first_loc = problems[0][0] if problems else (prefix or None)
    if len(problems) == 1:
        # The error's field already names the location; don't repeat it.
        return SignalConfigError(problems[0][1], field=first_loc or None)
    lines = "; ".join(f"{loc or '<spec>'}: {msg}" for loc, msg in problems)
    return SignalConfigError(
        f"{len(problems)} problems: {lines}", field=first_loc or None
    )


def load_spec(source: SpecLike, *, base_dir: str | Path | None = None) -> SignalSpec:
    """Read and validate a SignalSpec.

    Args:
        source: A :class:`SignalSpec` (returned as is), a dict, a JSON string, or
            the path of a ``.json`` file.
        base_dir: Directory for relative ``$source`` paths. Defaults to the spec
            file's directory, or the working directory for a dict or string.

    Returns:
        The validated spec, migrated to the current schema version.

    Raises:
        SignalConfigError: If the spec is invalid. ``field`` names the offending
            field, e.g. ``steps.2.window``.
    """
    if isinstance(source, SignalSpec):
        return source
    if isinstance(source, Mapping):
        data = json.loads(json.dumps(source))
        root = Path(base_dir) if base_dir is not None else Path.cwd()
    elif isinstance(source, str) and source.lstrip().startswith("{"):
        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            raise SignalConfigError(f"the spec is not valid JSON: {exc}") from None
        root = Path(base_dir) if base_dir is not None else Path.cwd()
    else:
        path = Path(source)
        data = _read_json(path)
        root = Path(base_dir) if base_dir is not None else path.parent
    if not isinstance(data, dict):
        raise SignalConfigError("a spec must be a JSON object.")

    data = migrate(resolve_sources(data, root))

    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raise SignalConfigError("steps must be a list.", field="steps")
    steps = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise SignalConfigError(
                "each step must be an object with a 'kind'.", field=f"steps.{i}.kind"
            )
        try:
            step_cls = get_step_class(raw["kind"])
        except SignalConfigError as exc:
            raise SignalConfigError(exc.message, field=f"steps.{i}.kind") from None
        try:
            steps.append(step_cls.model_validate(raw))
        except ValidationError as exc:
            raise _config_error(exc, f"steps.{i}") from None
    try:
        return SignalSpec.model_validate({**data, "steps": steps})
    except ValidationError as exc:
        raise _config_error(exc) from None


def _dump_model(model: BaseModel) -> dict[str, Any]:
    data = model.model_dump(mode="json", exclude_defaults=True)
    if "kind" in type(model).model_fields:
        data = {"kind": model.kind, **{k: v for k, v in data.items() if k != "kind"}}
    return data


def dump_spec(spec: SignalSpec, path: str | Path | None = None) -> dict[str, Any]:
    """Return a spec as a canonical JSON-able dict, optionally writing it to a file.

    Fields left at their defaults are omitted, except ``schema_version`` and each
    model's ``kind``, which are always written.

    Args:
        spec: The spec.
        path: If given, also write the spec there as indented JSON.

    Returns:
        The spec as a dict.
    """
    out: dict[str, Any] = {"schema_version": spec.schema_version, "name": spec.name}
    if spec.description is not None:
        out["description"] = spec.description
    if spec.inputs:
        out["inputs"] = [_dump_model(i) for i in spec.inputs]
    out["steps"] = [_dump_model(s) for s in spec.steps]
    if spec.outputs:
        out["outputs"] = [_dump_model(o) for o in spec.outputs]
    if spec.metadata:
        out["metadata"] = json.loads(json.dumps(spec.metadata))
    if path is not None:
        Path(path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def canonical_json(spec: SignalSpec) -> str:
    """Return the spec as compact, key-sorted JSON: the input to :func:`spec_hash`."""
    return json.dumps(dump_spec(spec), sort_keys=True, separators=(",", ":"))


def spec_hash(spec: SignalSpec) -> str:
    """Return the sha256 hex digest of the spec's canonical JSON.

    Two specs that describe the same pipeline have the same hash, whatever their
    key order, and whether defaults were written out or left implicit.
    """
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
