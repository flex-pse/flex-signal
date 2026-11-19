"""Scores for a model's predictions against what actually happened."""

from __future__ import annotations

import math

import pandas as pd

__all__ = ["metrics"]


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return a prediction's ``rmse``, ``mae`` and ``mape``.

    ``mape`` is the mean absolute error as a fraction of the actual value, over
    the non-zero actual values only; it is left out when every actual value is 0.

    Example:
        >>> import pandas as pd
        >>> from flexsignal.eval import metrics
        >>> metrics(pd.Series([0.0, 2.0, 4.0]), pd.Series([1.0, 2.0, 2.0]))["mae"]
        1.0
    """
    error = predicted - actual
    nonzero = actual != 0
    found = {
        "rmse": math.sqrt(float((error**2).mean())),
        "mae": float(error.abs().mean()),
    }
    if nonzero.any():
        found["mape"] = float((error[nonzero] / actual[nonzero]).abs().mean())
    return found
