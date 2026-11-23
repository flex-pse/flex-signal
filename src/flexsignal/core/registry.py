"""The step registry: maps a spec ``kind`` to the step class that implements it.

Built-in steps register themselves when :mod:`flexsignal.steps` is imported,
which ``import flexsignal`` always does. Third-party packages add steps through
the ``flexsignal.steps`` entry-point group. Each entry point names either a step
class or a module that registers its steps on import. One that fails to load
is skipped with a warning, so it can't hide the others.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from flexsignal.core.exceptions import SignalConfigError, SignalWarning

STEP_ENTRY_POINT_GROUP = "flexsignal.steps"
"""Entry-point group through which other packages register step classes."""

_STEPS: dict[str, type] = {}
_entry_points_loaded = False


def register_step(cls: type, *, replace: bool = False) -> type:
    """Register a step class under its ``kind``; usable as a class decorator.

    Args:
        cls: A step class whose ``kind`` field has a string default.
        replace: Replace a class already registered for the kind instead of
            raising. :func:`flexsignal.register_cleaner` and its siblings use it
            when the same function is registered again, e.g. by a re-run
            notebook cell.

    Returns:
        ``cls`` unchanged.

    Raises:
        SignalConfigError: If another class is already registered for the kind.
    """
    field = getattr(cls, "model_fields", {}).get("kind")
    kind = getattr(field, "default", None)
    if not isinstance(kind, str) or not kind:
        raise SignalConfigError(
            f"{cls.__name__} can't be registered: its 'kind' field needs a string "
            "default, e.g. kind: Literal['my_step'] = 'my_step'."
        )
    existing = _STEPS.get(kind)
    if existing is not None and existing is not cls and not replace:
        raise SignalConfigError(
            f"step kind {kind!r} is already registered by "
            f"{existing.__module__}.{existing.__qualname__}; pick a different kind."
        )
    _STEPS[kind] = cls
    return cls


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in entry_points(group=STEP_ENTRY_POINT_GROUP):
        # One broken plugin must not hide the others, or every later lookup.
        try:
            loaded = ep.load()
            if isinstance(loaded, type):
                register_step(loaded)
        except Exception as exc:  # a plugin can fail in any way on import
            target = getattr(ep, "value", None) or ep.name
            warnings.warn(
                f"could not load the {STEP_ENTRY_POINT_GROUP!r} entry point "
                f"{ep.name!r} ({target}): {type(exc).__name__}: {exc}. The steps it "
                "provides are unavailable; fix or reinstall the package that "
                "provides it, or uninstall it.",
                SignalWarning,
                stacklevel=3,
            )


def get_step_class(kind: str) -> type:
    """Return the step class registered for ``kind``.

    Raises:
        SignalConfigError: If no step is registered for ``kind``.
    """
    if kind not in _STEPS:
        _load_entry_points()
    try:
        return _STEPS[kind]
    except KeyError:
        known = ", ".join(sorted(_STEPS))
        if not kind.isidentifier():
            raise SignalConfigError(
                f"unknown step kind {kind!r}. A spec names a step by its kind, e.g. "
                "hampel, or a registered method's name, never an import path: "
                "register the method in code first (fs.register_cleaner and its "
                f"siblings). Known kinds: {known}.",
                value=kind,
            ) from None
        raise SignalConfigError(
            f"unknown step kind {kind!r}; register it first, e.g. "
            f"fs.register_imputer({kind!r}, fn) (or register_cleaner or "
            "register_forecaster), or install the package that provides it. "
            f"Known kinds: {known}.",
            value=kind,
        ) from None


def own_steps() -> dict[str, type]:
    """Return flex-signal's own steps: built in, or its adapters' registered methods.

    Steps from users, other packages and test modules are left out. This is what
    the committed JSON Schema and the reference tables describe.
    """
    return {
        kind: cls
        for kind, cls in registered_steps().items()
        if cls.__module__.split(".")[0] == "flexsignal"
        and ".tests" not in cls.__module__
    }


def find_step(kind: str) -> type | None:
    """Return the class registered for ``kind``, or None, without loading plugins."""
    return _STEPS.get(kind)


def unregister_step(kind: str) -> None:
    """Remove a registered kind if it is registered (for tests and notebooks)."""
    _STEPS.pop(kind, None)


def load_entry_points() -> None:
    """Load the steps that installed packages provide, once.

    An entry point that fails to load is skipped with a :class:`SignalWarning`
    naming it and the error; the others still load.
    """
    _load_entry_points()


def registered_steps() -> dict[str, type]:
    """Return every registered step class keyed by kind, sorted by kind."""
    _load_entry_points()
    return dict(sorted(_STEPS.items()))
