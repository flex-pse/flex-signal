Steps
=====

.. automodule:: flexsignal.steps
   :no-members:

.. flexsignal-step-table::

The cleaning, imputation and forecasting methods, with links to the libraries
behind them, are listed in :doc:`methods`.

Select
------

.. automodule:: flexsignal.steps.select
   :no-members:

.. currentmodule:: flexsignal.steps.select

.. autoclass:: Align
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Align

.. autoclass:: Resample
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Resample

.. autoclass:: Window
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Window

.. autoclass:: Mask
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Mask

.. autoclass:: Rename
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Rename

.. autoclass:: ConvertUnits
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.ConvertUnits

.. autoclass:: Scale
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Scale

.. autoclass:: Combine
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Combine

.. autoclass:: Shift
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.select.Shift

Clean
-----

.. automodule:: flexsignal.steps.clean
   :no-members:

.. currentmodule:: flexsignal.steps.clean

.. autoclass:: Range
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Range

.. autoclass:: RateOfChange
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.RateOfChange

.. note::

   The first valid value is always accepted, and the values after it are
   compared with it. If that first value is itself bad (a spike or a startup
   reading), the good values after it are rejected until enough time has passed
   for the allowed rate to reach them, which can be most of the data. Put a
   ``range`` or ``hampel`` step before ``rate_of_change`` so that a bad first
   value is removed before it can become the reference.

.. autoclass:: Stuck
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Stuck

.. autoclass:: Hampel
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Hampel

.. autoclass:: Tukey
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Tukey

.. autoclass:: Expression
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Expression

.. autoclass:: Filter
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Filter

.. autoclass:: Detotalize
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.clean.Detotalize

.. note::

   ``rollover`` is the counter's modulus, its maximum reading + 1: for a 6-digit
   counter that reads 0 to 999999, set ``rollover=1000000``. (The "e.g. 999999"
   in the description above is the counter's maximum reading, which is one
   less than ``rollover``.)

Correct
-------

.. automodule:: flexsignal.steps.correct
   :no-members:

.. currentmodule:: flexsignal.steps.correct

.. autoclass:: Calibrate
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.correct.Calibrate

.. autoclass:: RollingRefit
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.correct.RollingRefit

Impute
------

.. automodule:: flexsignal.steps.impute
   :no-members:

.. currentmodule:: flexsignal.steps.impute

.. autoclass:: Interpolate
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.Interpolate

.. autoclass:: Kalman
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.Kalman

.. autoclass:: Regression
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.Regression

.. autoclass:: CycleTerm
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.CycleTerm

.. autoclass:: KNN
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.KNN

.. autoclass:: Iterative
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.Iterative

.. autoclass:: Tiered
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.Tiered

.. autoclass:: ImputeTier
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.impute.ImputeTier

Forecast
--------

.. automodule:: flexsignal.steps.forecast
   :no-members:

.. currentmodule:: flexsignal.steps.forecast

.. autoclass:: Persistence
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.forecast.Persistence

.. autoclass:: SeasonalNaive
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.forecast.SeasonalNaive

.. autoclass:: Structural
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.forecast.Structural

.. autoclass:: Regression
   :no-members:

.. flexsignal-spec-table:: flexsignal.steps.forecast.Regression

.. autoclass:: ForecastStep
   :no-members:

.. autofunction:: quantile_column

Adapters
--------

.. automodule:: flexsignal.adapters
   :no-members:

Each adapter is a registered method: its options come from its function's signature, and the library is
imported only when the step runs. The extra in each heading installs it.

Anomaly detection (``[anomaly]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.anomaly
   :no-members:

.. py:class:: PyodDetector
   :module: flexsignal.steps.clean

   A registered method: ``fs.clean.PyodDetector``, or ``"kind": "PyodDetector"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.clean.PyodDetector

.. py:class:: RupturesChangepoint
   :module: flexsignal.steps.clean

   A registered method: ``fs.clean.RupturesChangepoint``, or ``"kind": "RupturesChangepoint"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.clean.RupturesChangepoint

.. py:class:: StumpyDiscord
   :module: flexsignal.steps.clean

   A registered method: ``fs.clean.StumpyDiscord``, or ``"kind": "StumpyDiscord"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.clean.StumpyDiscord

PyPOTS (``[pypots]``)
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.pypots
   :no-members:

.. autofunction:: flexsignal.adapters.pypots.to_windows

.. autofunction:: flexsignal.adapters.pypots.from_windows

.. autofunction:: flexsignal.adapters.pypots.window_starts

.. py:class:: PypotsImpute
   :module: flexsignal.steps.impute

   A registered method: ``fs.impute.PypotsImpute``, or ``"kind": "PypotsImpute"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.impute.PypotsImpute

.. py:class:: PypotsForecast
   :module: flexsignal.steps.forecast

   A registered method: ``fs.forecast.PypotsForecast``, or ``"kind": "PypotsForecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.PypotsForecast

Foundation models (``[foundation]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.foundation
   :no-members:

.. py:class:: ChronosForecast
   :module: flexsignal.steps.forecast

   A registered method: ``fs.forecast.ChronosForecast``, or ``"kind": "ChronosForecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.ChronosForecast

.. py:class:: T0Forecast
   :module: flexsignal.steps.forecast

   A registered method, in the ``[t0]`` extra: ``fs.forecast.T0Forecast``, or ``"kind": "T0Forecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.T0Forecast

neuralforecast (``[deep]``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.deep
   :no-members:

.. py:class:: NeuralforecastForecast
   :module: flexsignal.steps.forecast

   A registered method: ``fs.forecast.NeuralforecastForecast``, or ``"kind": "NeuralforecastForecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.NeuralforecastForecast

darts (``[darts]``)
^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.darts
   :no-members:

.. py:class:: DartsForecast
   :module: flexsignal.steps.forecast

   A registered method: ``fs.forecast.DartsForecast``, or ``"kind": "DartsForecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.DartsForecast

sktime (``[sktime]``)
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: flexsignal.adapters.sktime
   :no-members:

.. py:class:: SktimeForecast
   :module: flexsignal.steps.forecast

   A registered method: ``fs.forecast.SktimeForecast``, or ``"kind": "SktimeForecast"`` in a spec.

.. flexsignal-spec-table:: flexsignal.steps.forecast.SktimeForecast

Your own methods
----------------

.. automodule:: flexsignal.steps.registered
   :no-members:

.. currentmodule:: flexsignal

.. autofunction:: register_cleaner

.. autofunction:: register_imputer

.. autofunction:: register_forecaster

.. autofunction:: register_model

.. autofunction:: load_plugins

.. currentmodule:: flexsignal.steps.registered

.. autofunction:: registered_methods

.. autoclass:: Method
   :members:

.. autodata:: CLEANER_FLAGS

Writing a step
--------------

.. currentmodule:: flexsignal.steps

.. autoclass:: StreamStep
   :no-members:
