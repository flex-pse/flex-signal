"""Pipeline steps, grouped by what they do.

- :mod:`~flexsignal.steps.select`: put streams on a grid, choose windows and
  rows, reshape.
- :mod:`~flexsignal.steps.clean`: find bad values, set them to NaN, and flag why.
- :mod:`~flexsignal.steps.correct`: correct a stream against a trusted reference.
- :mod:`~flexsignal.steps.impute`: fill missing values and flag them as filled.
- :mod:`~flexsignal.steps.forecast`: predict future values and flag them as
  forecast.
- :mod:`~flexsignal.steps.registered`: your own cleaners, imputers and
  forecasters, registered from plain functions.

Every step is a frozen pydantic model, so the object you build in Python is
exactly what a JSON spec loads into. Importing this package registers the
built-in steps.
"""

from flexsignal.steps import clean, correct, forecast, impute, registered, select
from flexsignal.steps._base import StreamStep

__all__ = [
    "StreamStep",
    "clean",
    "correct",
    "forecast",
    "impute",
    "registered",
    "select",
]
