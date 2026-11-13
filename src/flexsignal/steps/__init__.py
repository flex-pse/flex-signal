"""Pipeline steps, grouped by what they do.

- :mod:`~flexsignal.steps.select`: put streams on a grid, choose windows and
  rows, reshape.
- :mod:`~flexsignal.steps.clean`: find bad values, set them to NaN, and flag why.
- :mod:`~flexsignal.steps.impute`: fill missing values and flag them as filled.

Every step is a frozen pydantic model, so the object you build in Python is
exactly what a JSON spec loads into. Importing this package registers the
built-in steps. Later milestones add correct, forecast and derive steps.
"""

from flexsignal.steps import clean, impute, select
from flexsignal.steps._base import StreamStep

__all__ = ["StreamStep", "clean", "impute", "select"]
