flex-signal
===========

flex-signal selects, cleans, corrects, imputes and forecasts plant timeseries
data for the `flex-pse <https://flex-pse.github.io/flex-pse/>`_ ecosystem.
It wraps established libraries such as pandas, scipy, scikit-learn,
statsmodels and PyPOTS behind a declarative, versioned pipeline spec. Its
results go straight into flex-pse models, ``flexparameterize`` surrogate
fits, and flex-orchestrator studies.

.. note::

   flex-signal is pre-alpha. Pipelines select, clean, calibrate against a
   reference, impute gaps and forecast, from Python or JSON specs, and fitted
   pipelines save as plain JSON. Forecasts are backtested against baselines.
   Methods of your own, and adapters to deep, foundation and anomaly-detection
   libraries, run as steps too.
   Cleaned and forecast data goes into flex-pse models on their own time steps,
   and is prepared for fitting surrogates. The API may still change before 1.0.

Start with :doc:`getting_started/ten_minutes` or the :doc:`tutorials/index`, or
go straight to :doc:`how_to/clean_a_stream`, then
:doc:`explanation/signal_frame_and_flags`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/ten_minutes
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/index

.. toctree::
   :maxdepth: 2
   :caption: How-to

   how_to/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Release notes

   changelog

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/methods
   reference/index
   reference/glossary
