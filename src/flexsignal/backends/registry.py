"""The backend registry: string keys that name an object in an existing library.

A step names the algorithm it uses by a key such as ``"scipy:PchipInterpolator"``
or ``"sklearn:HuberRegressor"``. The target is imported only when the key is
used, so an optional library costs nothing until a step needs it. If the library
is missing, the error says which extra installs it.

Besides registered keys, any ``"package.module:attribute"`` string resolves
directly, which is how user code plugs in. Third-party packages register keys
through the ``flexsignal.backends`` entry-point group. Each entry point names a
callable that is called with no arguments and registers its keys.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from flexsignal.core.exceptions import SignalBackendError

BACKEND_ENTRY_POINT_GROUP = "flexsignal.backends"
"""Entry-point group through which other packages register backends."""


@dataclass(frozen=True)
class BackendInfo:
    """A registered backend key."""

    key: str
    """The key steps use, e.g. ``"scipy:PchipInterpolator"``."""
    target: str
    """What it resolves to, as ``"package.module:attribute"``."""
    extra: str | None = None
    """The flex-signal extra that installs it, or None for core dependencies."""
    description: str = ""
    """One line on what it is."""
    docs_url: str | None = None
    """Upstream documentation."""


_BACKENDS: dict[str, BackendInfo] = {}
_entry_points_loaded = False


def register_backend(
    key: str,
    target: str,
    *,
    extra: str | None = None,
    description: str = "",
    docs_url: str | None = None,
) -> BackendInfo:
    """Register a backend key.

    Args:
        key: The key steps use, as ``"library:Name"``.
        target: What it resolves to, as ``"package.module:attribute"``.
        extra: The flex-signal extra that installs the library, if it is optional.
        description: One line on what it is.
        docs_url: Link to the upstream documentation.

    Returns:
        The registered :class:`BackendInfo`.

    Raises:
        SignalBackendError: If ``key`` is already registered with another target.
    """
    info = BackendInfo(key, target, extra, description, docs_url)
    existing = _BACKENDS.get(key)
    if existing is not None and existing.target != target:
        raise SignalBackendError(
            f"backend {key!r} is already registered for {existing.target}; "
            "pick another key."
        )
    _BACKENDS[key] = info
    return info


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in entry_points(group=BACKEND_ENTRY_POINT_GROUP):
        ep.load()()


def _import_target(target: str) -> Any:
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def get_backend(key: str) -> Any:
    """Return the object a backend key names, importing it on first use.

    Args:
        key: A registered key, or any ``"package.module:attribute"`` string.

    Returns:
        The imported object (a class, function or module attribute).

    Raises:
        SignalBackendError: If the key is unknown, or its library is not installed.
    """
    if key not in _BACKENDS:
        _load_entry_points()
    info = _BACKENDS.get(key)
    target = info.target if info else key
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        known = ", ".join(sorted(_BACKENDS))
        raise SignalBackendError(
            f"unknown backend {key!r}. Use a registered key ({known}) or a "
            "'package.module:attribute' path.",
            value=key,
        )
    try:
        return _import_target(target)
    except ImportError as exc:
        if info is not None and info.extra:
            hint = f"install it with `pip install flex-signal[{info.extra}]`"
        else:
            hint = f"install the package that provides {module_name!r}"
        raise SignalBackendError(
            f"backend {key!r} needs {module_name!r}, which is not installed: {hint}.",
            value=key,
        ) from exc
    except AttributeError as exc:
        raise SignalBackendError(
            f"backend {key!r}: {module_name!r} has no attribute {attr!r}.", value=key
        ) from exc


def registered_backends() -> list[BackendInfo]:
    """Return every registered backend, sorted by key."""
    _load_entry_points()
    return [_BACKENDS[k] for k in sorted(_BACKENDS)]
