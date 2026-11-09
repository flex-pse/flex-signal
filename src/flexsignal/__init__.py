"""flex-signal: timeseries selection, cleaning, imputation and forecasting.

flex-signal is the data layer of the flex-pse ecosystem. It wraps existing
libraries (pandas, scipy, scikit-learn, statsmodels, PyPOTS, ...) behind a
declarative, serializable pipeline spec, and hands its results to flex-pse
models, flexparameterize and flex-orchestrator studies.

Importing this package never imports flex-pse, pyomo, torch or polars. Those
are reached only through optional extras, lazily.
"""

from importlib.metadata import version as _dist_version

__version__ = _dist_version("flex-signal")

__all__ = ["__version__"]
