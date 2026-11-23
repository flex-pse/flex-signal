Files and backends
==================

File I/O
--------

.. automodule:: flexsignal.io.files
   :no-members:

.. currentmodule:: flexsignal.io

.. autofunction:: read_parquet

.. autofunction:: write_parquet

.. autofunction:: read_csv

.. autofunction:: write_csv

Backends
--------

.. automodule:: flexsignal.backends.registry
   :no-members:

.. flexsignal-backend-table::

.. currentmodule:: flexsignal.backends

.. autofunction:: get_backend

.. autofunction:: register_backend

.. autofunction:: registered_backends

.. autoclass:: BackendInfo
   :members:

.. automodule:: flexsignal.backends.builtin
   :no-members:

Models for calibrate and regression
-----------------------------------

.. automodule:: flexsignal.backends.models
   :no-members:

.. currentmodule:: flexsignal.backends.models

.. autodata:: LINEAR_MODELS
   :no-value:

.. autofunction:: fit_linear

.. autofunction:: predict_linear

.. autofunction:: fit_curve
