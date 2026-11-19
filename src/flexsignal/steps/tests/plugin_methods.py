"""A plugin module for the extension-API tests: it registers one method per role.

Tests load it with ``fs.load_plugins("flexsignal.steps.tests.plugin_methods")``
and unregister its names afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flexsignal as fs

NAMES = ("TestDespike", "TestNeighbourFill", "TestDrift", "TestScaledOLS")


def despike(values: pd.Series, state: dict, *, k: float = 4.0) -> pd.Series:
    """Reject values far from the training median.

    Args:
        values: The stream.
        state: The training median and spread.
        k: How many spreads from the median count as a spike.
    """
    return (values - state["median"]).abs() > k * state["spread"]


def fit_despike(values: pd.Series, *, k: float = 4.0) -> dict:
    median = float(values.median())
    return {"median": median, "spread": float((values - median).abs().median())}


def neighbour_fill(
    values: pd.DataFrame,
    gaps: pd.Series,
    state: dict,
    *,
    features: list[str],
    weight: float = 1.0,
) -> np.ndarray:
    """Fill a gap from other streams, scaled by the training ratio.

    Args:
        values: The streams, the one being filled first.
        gaps: Where to fill.
        state: The training ratio.
        features: The streams to fill from.
        weight: Scales the result.
    """
    return weight * state["ratio"] * values[features].mean(axis=1).to_numpy()


def fit_neighbour_fill(values: pd.DataFrame, *, features: list[str]) -> dict:
    target = values.iloc[:, 0]
    source = values[features].mean(axis=1)
    rows = target.notna() & source.notna()
    return {"ratio": float(target[rows].sum() / source[rows].sum())}


def drift(history: pd.DataFrame, future: pd.DataFrame, *, spread: float = 1.0):
    """Hold the last value plus the average step; a fixed spread."""
    y = history.iloc[:, 0].dropna()
    slope = float(np.diff(y.to_numpy()).mean())
    mean = y.iloc[-1] + slope * np.arange(1, len(future) + 1)
    return {"mean": mean, "std": np.full(len(future), spread)}


class ScaledOLS:
    """Least squares on inputs scaled by 2, so its coefficients differ from OLS."""

    def __init__(self, scale: float = 2.0):
        self.scale = scale

    def fit(self, X, y):
        design = np.column_stack([np.ones(len(y)), np.asarray(X) * self.scale])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.intercept_ = coef[0]
        self.coef_ = coef[1:] * self.scale
        return self


fs.register_cleaner("TestDespike", despike, fit=fit_despike, flag="OUTLIER")
fs.register_imputer("TestNeighbourFill", neighbour_fill, fit=fit_neighbour_fill)
fs.register_forecaster("TestDrift", drift)
fs.register_model("TestScaledOLS", ScaledOLS, description="OLS on scaled inputs.")
