"""The step registry: maps a spec ``kind`` to the step class that implements it.

Built-in steps register themselves when :mod:`flexsignal.steps` is imported,
which ``import flexsignal`` always does. Third-party packages add steps through
the ``flexsignal.steps`` entry-point group. Each entry point names either a step
class or a module that registers its steps on import.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from flexsignal.core.exceptions import SignalConfigError

STEP_ENTRY_POINT_GROUP = "flexsignal.steps"
"""Entry-point group through which other packages register step classes."""

_STEPS: dict[str, type] = {}
_entry_points_loaded = False


def register_step(cls: type) -> type:
    """Register a step class under its ``kind``; usable as a class decorator.

    Args:
        cls: A step class whose ``kind`` field has a string default.

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
    if existing is not None and existing is not cls:
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
        loaded = ep.load()
        if isinstance(loaded, type):
            register_step(loaded)


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
        raise SignalConfigError(
            f"unknown step kind {kind!r}. Known kinds: {known}. A step from another "
            f"package must be installed and registered through the "
            f"{STEP_ENTRY_POINT_GROUP!r} entry-point group.",
            value=kind,
        ) from None


def registered_steps() -> dict[str, type]:
    """Return every registered step class keyed by kind, sorted by kind."""
    _load_entry_points()
    return dict(sorted(_STEPS.items()))
