"""Unit conversion at the boundary, with pint (the ``[units]`` extra).

Units are metadata: values stay plain float64 and every step works on them
unchanged. Conversion happens only where it is asked for, a ``convert_units``
step or a ``range`` bound given in other units, and only then is pint imported.
Units are written in the pyomo/pint syntax flex-pse uses (``"m^3/hr"``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from flexsignal.core.exceptions import SignalBackendError, SignalConfigError

__all__ = ["convert", "same_units"]


@lru_cache(maxsize=1)
def _registry() -> Any:
    try:
        import pint
    except ImportError as exc:
        raise SignalBackendError(
            "unit conversion needs pint, which is not installed: install it with "
            "`pip install flex-signal[units]`."
        ) from exc
    return pint.UnitRegistry()


def _units(registry: Any, units: str) -> Any:
    try:
        return registry.Unit(units)
    except Exception as exc:  # pint raises several error types for bad units
        raise SignalConfigError(
            f"pint can't read the units {units!r}: {exc}. Use the pyomo/pint syntax, "
            "e.g. m^3/hr or kWh.",
            value=units,
        ) from None


def convert(values: Any, from_units: str, to_units: str) -> np.ndarray | float:
    """Convert values from one unit to another.

    Offset units such as ``degC`` and ``K`` convert correctly. NaN stays NaN.

    Args:
        values: A number or an array of numbers.
        from_units: Their units, e.g. ``"m^3/hr"``.
        to_units: The units wanted, e.g. ``"m^3/s"``.

    Returns:
        A float for a number, else a float64 array.

    Raises:
        SignalBackendError: If pint isn't installed.
        SignalConfigError: If either unit can't be read, or they measure
            different things (``"m^3/hr"`` to ``"kW"``).

    Examples:
        >>> round(convert(3600.0, "m^3/hr", "m^3/s"), 6)
        1.0
    """
    registry = _registry()
    source = _units(registry, from_units)
    target = _units(registry, to_units)
    scalar = np.ndim(values) == 0
    magnitude = np.asarray(values, dtype="float64")
    try:
        out = registry.Quantity(magnitude, source).to(target).magnitude
    except Exception as exc:  # pint's DimensionalityError and friends
        raise SignalConfigError(
            f"can't convert {from_units!r} to {to_units!r}: {exc}.", value=to_units
        ) from None
    return float(out) if scalar else np.asarray(out, dtype="float64")


def same_units(a: str | None, b: str | None) -> bool:
    """Return True if two unit strings are written the same way (no conversion)."""
    return (a or None) == (b or None)
