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

# Imported as modules: fs.handoff.to_grid, fs.eval.benchmark_imputers.
# flex-signal's adapters to optional libraries register their methods here;
# the libraries themselves are imported only when a step runs.
from flexsignal import (  # noqa: A004  (eval: the subpackage)
    adapters,  # noqa: E402,F401  (registers the adapters)
    eval,
    handoff,
)
from flexsignal.api import RunResult, fit, run
from flexsignal.backends.models import register_model
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
from flexsignal.eval import backtest
from flexsignal.pipeline import FittedPipeline, Pipeline
from flexsignal.spec import SCHEMA_VERSION, SignalSpec, dump_spec, load_spec, spec_hash
from flexsignal.steps import clean, correct, forecast, impute, select
from flexsignal.steps.registered import (
    load_plugins,
    register_cleaner,
    register_forecaster,
    register_imputer,
)

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
    "backtest",
    "clean",
    "correct",
    "eval",
    "dump_spec",
    "fit",
    "forecast",
    "handoff",
    "impute",
    "load_plugins",
    "load_spec",
    "local_timezone",
    "register_cleaner",
    "register_forecaster",
    "register_imputer",
    "register_model",
    "run",
    "select",
    "spec_hash",
]
