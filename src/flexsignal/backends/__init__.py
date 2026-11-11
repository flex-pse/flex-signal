"""Lazy adapters from a string key (``"scipy:PchipInterpolator"``) to a library.

Steps name the library object they use by key. The object is imported only
when used, so optional libraries cost nothing until a step needs them. M1
registers the scipy and numpy interpolators that ``interpolate`` uses. Later
milestones add sklearn, statsmodels, PyPOTS and others.
"""

from flexsignal.backends.registry import (
    BACKEND_ENTRY_POINT_GROUP,
    BackendInfo,
    get_backend,
    register_backend,
    registered_backends,
)

_SCIPY_DOCS = "https://docs.scipy.org/doc/scipy/reference/generated/"

register_backend(
    "numpy:interp",
    "numpy:interp",
    description="Piecewise-linear interpolation.",
    docs_url="https://numpy.org/doc/stable/reference/generated/numpy.interp.html",
)
register_backend(
    "scipy:interp1d",
    "scipy.interpolate:interp1d",
    description="Nearest-neighbour and previous-value interpolation.",
    docs_url=f"{_SCIPY_DOCS}scipy.interpolate.interp1d.html",
)
register_backend(
    "scipy:PchipInterpolator",
    "scipy.interpolate:PchipInterpolator",
    description="Shape-preserving piecewise-cubic (PCHIP) interpolation; no overshoot.",
    docs_url=f"{_SCIPY_DOCS}scipy.interpolate.PchipInterpolator.html",
)
register_backend(
    "scipy:Akima1DInterpolator",
    "scipy.interpolate:Akima1DInterpolator",
    description="Akima piecewise-cubic interpolation; robust to outliers in shape.",
    docs_url=f"{_SCIPY_DOCS}scipy.interpolate.Akima1DInterpolator.html",
)

__all__ = [
    "BACKEND_ENTRY_POINT_GROUP",
    "BackendInfo",
    "get_backend",
    "register_backend",
    "registered_backends",
]
