"""Evaluation: a masking benchmark for imputers, and forecast backtests.

:func:`benchmark_imputers` hides known values, lets each imputer fill them, and
scores the fill by gap length. :func:`backtest` issues a pipeline's forecast at
several times in the data's past and scores it, against baselines if asked.
:func:`metrics` scores any prediction (RMSE, MAE, MAPE).
"""

from flexsignal.eval.backtesting import BacktestResult, backtest
from flexsignal.eval.imputation import benchmark_imputers, hide_values, summarize
from flexsignal.eval.scores import metrics

__all__ = [
    "BacktestResult",
    "backtest",
    "benchmark_imputers",
    "hide_values",
    "metrics",
    "summarize",
]
