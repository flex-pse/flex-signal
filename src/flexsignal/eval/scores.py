"""Scores for a model's predictions against what actually happened."""

from __future__ import annotations

import math

import pandas as pd

__all__ = ["metrics"]


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return a prediction's ``rmse``, ``mae`` and ``mape``.

    ``mape`` is the mean absolute error as a fraction of the actual value, over
    the non-zero actual values only; it is left out when every actual value is 0.
    A pair with a missing value on either side isn't scored; with none left,
    every score is NaN.

    Example:
        >>> import pandas as pd
        >>> from flexsignal.eval import metrics
        >>> metrics(pd.Series([0.0, 2.0, 4.0]), pd.Series([1.0, 2.0, 2.0]))["mae"]
        1.0
    """
    actual = (
        actual.astype(float)
        if isinstance(actual, pd.Series)
        else pd.Series(actual, dtype=float)
    )
    predicted = (
        predicted.astype(float)
        if isinstance(predicted, pd.Series)
        else pd.Series(predicted, dtype=float)
    )
    actual, predicted = actual.align(predicted)  # pairs by index, as before
    both = actual.notna() & predicted.notna()
    actual, predicted = actual[both], predicted[both]
    if not len(actual):
        return {"rmse": math.nan, "mae": math.nan, "mape": math.nan}
    error = predicted - actual
    nonzero = actual != 0
    found = {
        "rmse": math.sqrt(float((error**2).mean())),
        "mae": float(error.abs().mean()),
    }
    if nonzero.any():
        found["mape"] = float((error[nonzero] / actual[nonzero]).abs().mean())
    return found
