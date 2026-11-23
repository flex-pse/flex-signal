Hand-off to models
==================

.. automodule:: flexsignal.handoff
   :no-members:

.. currentmodule:: flexsignal.handoff

Values per model time step
--------------------------

.. automodule:: flexsignal.handoff.grid
   :no-members:

.. currentmodule:: flexsignal.handoff

.. autofunction:: to_grid

.. autofunction:: data_window

.. autoclass:: GridValues
   :members: to_overrides, data_hash

.. autoclass:: GridOptions
   :no-members:

.. flexsignal-spec-table:: flexsignal.handoff.GridOptions

.. autoclass:: GridResample
   :no-members:

.. flexsignal-spec-table:: flexsignal.handoff.GridResample

.. autofunction:: values_hash

Models with several time blocks
-------------------------------

.. automodule:: flexsignal.handoff.scopes
   :no-members:

.. currentmodule:: flexsignal.handoff

.. autofunction:: grids_of

.. autofunction:: scope_of

.. autofunction:: targets

Fit data for a regressor
------------------------

.. automodule:: flexsignal.handoff.fit
   :no-members:

.. currentmodule:: flexsignal.handoff

.. autofunction:: fit_data

.. autoclass:: FitData
   :members: input_units, output_units

.. autoclass:: FitPrep
   :no-members:

.. flexsignal-spec-table:: flexsignal.handoff.FitPrep

.. autoclass:: FitResample
   :no-members:

.. flexsignal-spec-table:: flexsignal.handoff.FitResample

.. autofunction:: split_holdout

.. autofunction:: data_hash
