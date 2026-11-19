"""Hand data to models: values per model time step, and X, y for a regressor.

- :func:`to_grid` lines timeseries up with a model's time steps (facility time
  zone, interval resampling, gap rules, or one number with ``reduce``);
  :func:`data_window` says which data that reads.
- :func:`grids_of`, :func:`scope_of` and :func:`targets` decide which time block
  a model path belongs to, for models with several.
- :func:`fit_data` joins input and output data and prepares it for a regressor;
  :func:`split_holdout` keeps the end for testing.

Nothing here imports a model library. Turning values into flex-pse elements is
:mod:`flexsignal.integrations.flexpse`. These functions came from
flex-orchestrator, which now calls them.
"""

from flexsignal.handoff.fit import FitData, data_hash, fit_data, split_holdout
from flexsignal.handoff.grid import GridValues, data_window, to_grid, values_hash
from flexsignal.handoff.options import FitPrep, FitResample, GridOptions, GridResample
from flexsignal.handoff.scopes import grids_of, scope_of, targets

__all__ = [
    "FitData",
    "FitPrep",
    "FitResample",
    "GridOptions",
    "GridResample",
    "GridValues",
    "data_hash",
    "data_window",
    "fit_data",
    "grids_of",
    "scope_of",
    "split_holdout",
    "targets",
    "to_grid",
    "values_hash",
]
