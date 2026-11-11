"""flex-signal: timeseries selection, cleaning, imputation and forecasting.

flex-signal is the data layer of the flex-pse ecosystem. It wraps existing
libraries (pandas, scipy, scikit-learn, statsmodels, PyPOTS, ...) behind a
declarative, serializable pipeline spec. It hands its results to flex-pse
models, flexparameterize and flex-orchestrator studies.

Importing this package never imports flex-pse, pyomo, torch or polars. Those
are reached only through optional extras, lazily.

Example:
    >>> import flexsignal as fs
    >>> pipe = fs.Pipeline(
    ...     [
    ...         fs.clean.Range("flow", min=0),
    ...         fs.impute.Interpolate("flow", max_gap="15min"),
    ...     ],
    ...     name="flow_cleanup",
    ... )
"""

from importlib.metadata import version as _dist_version

from flexsignal.api import RunResult, fit, run
from flexsignal.core import (
    Flag,
    NaiveTimestampWarning,
    SignalConfigError,
    SignalDataError,
    SignalError,
    SignalFrame,
    StreamMeta,
    local_timezone,
)
from flexsignal.pipeline import FittedPipeline, Pipeline
from flexsignal.spec import SCHEMA_VERSION, SignalSpec, dump_spec, load_spec, spec_hash
from flexsignal.steps import clean, impute, select

__version__ = _dist_version("flex-signal")

__all__ = [
    "SCHEMA_VERSION",
    "FittedPipeline",
    "Flag",
    "NaiveTimestampWarning",
    "Pipeline",
    "RunResult",
    "SignalConfigError",
    "SignalDataError",
    "SignalError",
    "SignalFrame",
    "SignalSpec",
    "StreamMeta",
    "__version__",
    "clean",
    "dump_spec",
    "fit",
    "impute",
    "load_spec",
    "local_timezone",
    "run",
    "select",
    "spec_hash",
]
