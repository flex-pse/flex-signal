"""The backend keys flex-signal registers for its own steps.

Every built-in key is registered here, in one place. The registry loads this
module the first time a key is looked up, so importing ``flexsignal`` doesn't
import scipy, scikit-learn or statsmodels; each library is imported only when a
step uses one of its keys.

To add a key from your own package, call
:func:`~flexsignal.backends.register_backend` from a function exposed through
the ``flexsignal.backends`` entry-point group.
"""

from __future__ import annotations

from flexsignal.backends.models import CURVE_FIT, LINEAR_MODELS
from flexsignal.backends.registry import register_backend

_SCIPY_DOCS = "https://docs.scipy.org/doc/scipy/reference/generated/"
_SKLEARN_DOCS = "https://scikit-learn.org/stable/modules/generated/"
_SM_DOCS = "https://www.statsmodels.org/stable/generated/"

# -- interpolation (interpolate) -----------------------------------------------------

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

# -- filters (filter) ----------------------------------------------------------------

register_backend(
    "scipy:savgol_filter",
    "scipy.signal:savgol_filter",
    description="Savitzky-Golay smoothing: a moving polynomial fit.",
    docs_url=f"{_SCIPY_DOCS}scipy.signal.savgol_filter.html",
)
register_backend(
    "scipy:butter",
    "scipy.signal:butter",
    description="Butterworth filter design (second-order sections).",
    docs_url=f"{_SCIPY_DOCS}scipy.signal.butter.html",
)
register_backend(
    "scipy:sosfiltfilt",
    "scipy.signal:sosfiltfilt",
    description="Zero-phase (forward and backward) filtering.",
    docs_url=f"{_SCIPY_DOCS}scipy.signal.sosfiltfilt.html",
)

# -- imputers (kalman, knn, iterative) -----------------------------------------------

register_backend(
    "statsmodels:UnobservedComponents",
    "statsmodels.tsa.statespace.structural:UnobservedComponents",
    description="Structural state-space model; its Kalman smoother fills gaps.",
    docs_url=f"{_SM_DOCS}statsmodels.tsa.statespace.structural."
    "UnobservedComponents.html",
)
register_backend(
    "sklearn:KNNImputer",
    "sklearn.impute:KNNImputer",
    description="Fill a value from the k most similar rows.",
    docs_url=f"{_SKLEARN_DOCS}sklearn.impute.KNNImputer.html",
)
register_backend(
    "sklearn:IterativeImputer",
    "flexsignal.backends._iterative:IterativeImputer",
    description="Model each stream from the others, round-robin (MICE-style).",
    docs_url=f"{_SKLEARN_DOCS}sklearn.impute.IterativeImputer.html",
)

# -- models (calibrate, regression) ----------------------------------------------------

for _key, _line in LINEAR_MODELS.items():
    _lib, _name = _key.split(":")
    if _lib == "sklearn":
        register_backend(
            _key,
            f"sklearn.linear_model:{_name}",
            description=_line,
            docs_url=f"{_SKLEARN_DOCS}sklearn.linear_model.{_name}.html",
        )
    else:
        register_backend(
            _key,
            f"statsmodels.api:{_name}",
            description=_line,
            docs_url=(
                f"{_SM_DOCS}statsmodels.regression.linear_model.{_name}.html"
                if _name == "OLS"
                else f"{_SM_DOCS}statsmodels.robust.robust_linear_model.RLM.html"
            ),
        )
register_backend(
    CURVE_FIT,
    "scipy.optimize:curve_fit",
    description="Least-squares fit of a formula's parameters.",
    docs_url=f"{_SCIPY_DOCS}scipy.optimize.curve_fit.html",
)
