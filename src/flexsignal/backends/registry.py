"""The backend registry: string keys that name an object in an existing library.

A step names the algorithm it uses by a key such as ``"scipy:PchipInterpolator"``
or ``"sklearn:HuberRegressor"``. The target is imported only when the key is
used, so an optional library costs nothing until a step needs it. If the library
is missing, the error says which extra installs it.

flex-signal's own keys live in :mod:`flexsignal.backends.builtin`, which the
registry loads the first time it is used. Code of your own adds keys with
:func:`~flexsignal.backends.register_backend`. Only registered keys resolve: a
key is never treated as an import path, so naming a backend can't import
arbitrary code.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from flexsignal.core.exceptions import SignalBackendError


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
_builtins_loaded = False


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
        SignalBackendError: If ``target`` isn't ``"package.module:attribute"``,
            or ``key`` is already registered with another target.
    """
    parts = target.partition(":") if isinstance(target, str) else ("", "", "")
    module_name, sep, attr = parts
    if not (sep and module_name and attr):
        raise SignalBackendError(
            f"backend target {target!r} must look like 'package.module:attribute', "
            "e.g. 'scipy.interpolate:PchipInterpolator'.",
            value=target,
        )
    _load_builtins()
    info = BackendInfo(key, target, extra, description, docs_url)
    existing = _BACKENDS.get(key)
    if existing is not None and existing.target != target:
        raise SignalBackendError(
            f"backend {key!r} is already registered for {existing.target}; "
            "pick another key."
        )
    _BACKENDS[key] = info
    return info


def _load_builtins() -> None:
    """Register flex-signal's own keys (see :mod:`flexsignal.backends.builtin`)."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    importlib.import_module("flexsignal.backends.builtin")


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
        key: A registered key, e.g. ``"scipy:PchipInterpolator"``.

    Returns:
        The imported object (a class, function or module attribute).

    Raises:
        SignalBackendError: If the key is unknown, or its library is not installed.
    """
    _load_builtins()
    info = _BACKENDS.get(key)
    if info is None:
        known = ", ".join(sorted(_BACKENDS))
        raise SignalBackendError(
            f"unknown backend key {key!r}; register it first with "
            f"flexsignal.backends.register_backend. Known keys: {known}.",
            value=key,
        )
    target = info.target
    module_name, _, attr = target.partition(":")
    try:
        return _import_target(target)
    except ImportError as exc:
        if info.extra:
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
    _load_builtins()
    return [_BACKENDS[k] for k in sorted(_BACKENDS)]
