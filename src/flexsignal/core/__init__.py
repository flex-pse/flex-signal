"""Core data model: SignalFrame, quality flags, stream metadata, errors, registries.

The lowest layer. It depends only on pandas, numpy, pydantic and tzlocal, and every
other subpackage may import it.
"""

from flexsignal.core.conditions import (
    CONDITION_FUNCTIONS,
    check_condition,
    evaluate_condition,
)
from flexsignal.core.exceptions import (
    NaiveTimestampWarning,
    SignalBackendError,
    SignalConfigError,
    SignalDataError,
    SignalError,
    SignalWarning,
)
from flexsignal.core.flags import (
    FLAG_DESCRIPTIONS,
    FLAG_DTYPE,
    QUALITY_FLAGS,
    Flag,
    flag_names,
    parse_flags,
)
from flexsignal.core.frame import (
    FLAGS_SUFFIX,
    INDEX_NAME,
    SignalFrame,
    local_timezone,
)
from flexsignal.core.meta import StreamMeta
from flexsignal.core.registry import (
    STEP_ENTRY_POINT_GROUP,
    get_step_class,
    register_step,
    registered_steps,
)

__all__ = [
    "CONDITION_FUNCTIONS",
    "FLAGS_SUFFIX",
    "FLAG_DESCRIPTIONS",
    "FLAG_DTYPE",
    "INDEX_NAME",
    "QUALITY_FLAGS",
    "STEP_ENTRY_POINT_GROUP",
    "Flag",
    "NaiveTimestampWarning",
    "SignalBackendError",
    "SignalConfigError",
    "SignalDataError",
    "SignalError",
    "SignalFrame",
    "SignalWarning",
    "StreamMeta",
    "check_condition",
    "evaluate_condition",
    "flag_names",
    "get_step_class",
    "local_timezone",
    "parse_flags",
    "register_step",
    "registered_steps",
]
