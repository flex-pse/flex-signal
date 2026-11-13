"""The flex-signal error hierarchy.

Every error flex-signal raises subclasses :class:`SignalError`. A message says
what was wrong and what to do about it. Errors about a spec carry ``field``, the
dotted path of the offending spec field (for example ``steps.2.window``), so a
caller such as flex-orchestrator can point the user at it.

The names mirror flex-pse's ``flexcore.exceptions`` (``FlexError``,
``FlexConfigError``, ``FlexDataError``) without importing flex-pse.
"""

from __future__ import annotations

from typing import Any


class SignalError(Exception):
    """Base class for every error flex-signal raises.

    Args:
        message: What went wrong and what the user should do about it.
        field: Dotted path of the spec field the error is about, if any.
        value: The offending value, if useful to show.
    """

    def __init__(self, message: str, *, field: str | None = None, value: Any = None):
        """Store the message, field and value (see the class docstring)."""
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value

    def __str__(self) -> str:
        """Return the message, prefixed by the field path when there is one."""
        if self.field:
            return f"{self.field}: {self.message}"
        return self.message

    def __reduce__(self):
        """Pickle with the keyword arguments, so errors survive Ray and subprocesses."""
        return (_rebuild_error, (type(self), self.message, self.field, self.value))


def _rebuild_error(cls, message, field, value):
    return cls(message, field=field, value=value)


class SignalConfigError(SignalError, ValueError):
    """A spec, step option or argument is invalid."""


class SignalDataError(SignalError, ValueError):
    """Input data can't be used as given (index, dtype, gaps, ...)."""


class SignalBackendError(SignalConfigError):
    """A backend key is unknown, or its optional extra is not installed."""


class SignalWarning(UserWarning):
    """Something worked, but probably not the way the user intended."""


class NaiveTimestampWarning(SignalWarning):
    """Timestamps had no time zone, so they were taken as this machine's local time.

    Pass ``timezone=`` to say which zone the data is in, which also silences the
    warning. To silence it everywhere::

        warnings.filterwarnings("ignore", category=NaiveTimestampWarning)
    """
