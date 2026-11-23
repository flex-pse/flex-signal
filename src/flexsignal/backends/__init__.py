"""Lazy adapters from a string key (``"scipy:PchipInterpolator"``) to a library.

Steps name the library object they use by key. The object is imported only
when used, so optional libraries cost nothing until a step needs them.

- :mod:`~flexsignal.backends.registry`: registering and looking up keys.
- :mod:`~flexsignal.backends.builtin`: every key flex-signal registers itself.
- :mod:`~flexsignal.backends.models`: fitting and applying the linear models
  and formulas that ``calibrate`` and ``regression`` use.
"""

from flexsignal.backends import models
from flexsignal.backends.registry import (
    BackendInfo,
    get_backend,
    register_backend,
    registered_backends,
)

__all__ = [
    "BackendInfo",
    "get_backend",
    "models",
    "register_backend",
    "registered_backends",
]
