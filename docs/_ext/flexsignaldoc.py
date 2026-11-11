"""Sphinx directives that generate reference tables from flex-signal's own registries.

This is the counterpart of flex-pse's ``docs/_ext/flexdoc.py``. Generating
tables from the code means a reference page can never drift from what the
code actually registers. The directives arrive with the registries they read
(PLAN.md §7):

``.. flexsignal-step-table::`` (M1)
    Every registered step ``kind`` with its category, backing library,
    required extra and summary.

``.. flexsignal-spec-table:: <dotted pydantic model path>`` (M1)
    A step or spec model's fields: name, type, default, description.

``.. flexsignal-flag-table::`` (M1)
    The quality-flag bits and what sets each one.

``.. flexsignal-backend-table::`` (M1, grows to M5)
    Every backend key with its extra and upstream documentation link.

Each directive is a thin wrapper over a plain ``collect_*`` function with no
Sphinx dependency, so unit tests can call the collection logic directly.
"""

from __future__ import annotations


def setup(app):
    """Register the flex-signal directives (none until M1)."""
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
