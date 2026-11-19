"""Helpers the adapters share: optional libraries, model names, grids and seeds."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import Field

from flexsignal.core.exceptions import (
    SignalBackendError,
    SignalConfigError,
    SignalDataError,
)


def require(module: str, extra: str, *, method: str) -> ModuleType:
    """Import an optional library, or say which extra installs it.

    Args:
        module: The module to import, e.g. ``"pyod.models.ecod"``.
        extra: The flex-signal extra that installs it, e.g. ``"anomaly"``.
        method: The registered method that needs it, for the message.

    Returns:
        The imported module.

    Raises:
        SignalBackendError: If the library isn't installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SignalBackendError(
            f"{method} needs {module.split('.')[0]!r}, which is not installed: "
            f"install it with `pip install flex-signal[{extra}]`.",
            value=method,
        ) from exc


def regular_step(index: pd.DatetimeIndex, method: str) -> pd.Timedelta:
    """Return the single spacing of a regular index, or raise naming the method."""
    if len(index) < 2:
        raise SignalDataError(f"{method} needs at least two timestamps.")
    diffs = np.unique(np.diff(index.asi8))
    if len(diffs) != 1:
        raise SignalDataError(
            f"{method} needs a regular time index; put an align or resample step "
            "before it."
        )
    return pd.Timedelta(int(diffs[0]), unit="ns")


def steps_in(duration: str, step: pd.Timedelta, *, field: str, method: str) -> int:
    """Return how many data steps a duration option covers (at least 1)."""
    try:
        delta = pd.Timedelta(duration)
    except (TypeError, ValueError):
        raise SignalDataError(
            f"{method}: {field} {duration!r} isn't a duration, e.g. 1h."
        ) from None
    n, remainder = divmod(delta, step)
    if remainder != pd.Timedelta(0) or n < 1:
        raise SignalDataError(
            f"{method}: {field} {duration} isn't a whole number of the data's {step} "
            "steps."
        )
    return int(n)


def seed_torch(seed: int) -> None:
    """Seed numpy, Python and torch (if installed) so a training run repeats."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-free adapters
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# -- model names ----------------------------------------------------------------------

ModelName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
"""A library's model class, by bare name (e.g. ``SAITS``): never an import path."""


@dataclass(frozen=True)
class ModelOption:
    """An adapter option that names one of a library's models.

    The ``tested`` models run with the adapter's own simple settings; any other
    model class of the library runs too, untested, with its own settings in
    the step's ``params``. The methods reference page lists these.
    """

    method: str
    option: str
    library: str
    tested: tuple[str, ...]
    catalogue: str
    params: str = "params"
    note: str = ""


MODEL_OPTIONS: list[ModelOption] = []
"""Every adapter's model option, in registration order."""


def model_option(**kwargs: Any) -> ModelOption:
    """Record an adapter's model option for the docs, and return it."""
    option = ModelOption(**kwargs)
    MODEL_OPTIONS[:] = [o for o in MODEL_OPTIONS if o.method != option.method]
    MODEL_OPTIONS.append(option)
    return option


def model_class(
    module: Any,
    name: str,
    option: ModelOption,
    *,
    base: type | tuple[type, ...] | None = None,
) -> type:
    """Return the model class ``name`` from a library module.

    Only public classes of that module (subclasses of ``base``, or of one of
    them, if given) can be
    returned, so a spec can reach nothing but the library's models.

    Raises:
        SignalConfigError: If the library has no such model.
    """
    found = None if name.startswith("_") else getattr(module, name, None)
    if not isinstance(found, type) or (
        base is not None and not issubclass(found, base)
    ):
        raise SignalConfigError(
            f"{option.method}: {option.library} has no model {name!r}. Tested: "
            f"{', '.join(option.tested)}; any other {option.library} model works "
            f"too, by its class name: see {option.catalogue}.",
            field=option.option,
            value=name,
        )
    return found


def build_model(
    cls: type,
    settings: dict[str, Any],
    params: dict[str, Any],
    option: ModelOption,
    *,
    managed: frozenset[str] = frozenset(),
) -> Any:
    """Make a model from the adapter's ``settings`` and the user's ``params``.

    ``params`` may set any constructor argument except the ``managed`` ones,
    which the adapter sets itself. A required argument that neither gives is
    named in the error, so an untested model says what it needs.

    Raises:
        SignalConfigError: If ``params`` names an unknown or managed argument, a
            required argument is missing, or the library refuses the settings.
    """
    signature = inspect.signature(cls.__init__)
    accepted = {
        name
        for name, p in signature.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    takes_any = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    blocked = sorted(set(params) & managed)
    unknown = [] if takes_any else sorted(set(params) - accepted - managed)
    if blocked or unknown:
        raise SignalConfigError(
            f"{option.method}: {option.params} "
            + (
                f"{blocked} are set by flex-signal: use the step's own options"
                if blocked
                else ""
            )
            + ("; " if blocked and unknown else "")
            + (
                f"{unknown} aren't arguments of {option.library} {cls.__name__}"
                if unknown
                else ""
            )
            + f". {cls.__name__} accepts {sorted(accepted - managed)}.",
            field=option.params,
            value=blocked + unknown,
        )
    # The adapter's settings fit its tested models; another model takes the
    # ones its constructor has.
    settings = {k: v for k, v in settings.items() if takes_any or k in accepted}
    kwargs = {**settings, **params}
    missing = [
        name
        for name, p in signature.parameters.items()
        if name != "self"
        and p.default is p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and name not in kwargs
    ]
    if missing:
        tested = cls.__name__ in option.tested
        raise SignalConfigError(
            f"{option.method}: {option.library} {cls.__name__} needs {missing}"
            + (
                ""
                if tested
                else f", which flex-signal doesn't set for untested models: give "
                f"them in {option.params}, e.g. {option.params}="
                f"{{{missing[0]!r}: ...}}. See {option.catalogue}"
            )
            + ".",
            field=option.params,
            value=missing,
        )
    try:
        return cls(**kwargs)
    except (TypeError, ValueError, AssertionError) as exc:
        raise SignalConfigError(
            f"{option.method}: {option.library} {cls.__name__} refused these "
            f"settings: {exc}",
            field=option.params,
        ) from None
