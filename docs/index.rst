flex-signal
===========

flex-signal selects, cleans, corrects, imputes and forecasts plant timeseries
data for the `flex-pse <https://flex-pse.github.io/flex-pse/>`_ ecosystem.
It wraps established libraries such as pandas, scipy, scikit-learn,
statsmodels and PyPOTS behind a declarative, versioned pipeline spec. Its
results go straight into flex-pse models, ``flexparameterize`` surrogate
fits, and flex-orchestrator studies.

.. note::

   flex-signal is pre-alpha (M1). Cleaning pipelines work: select, clean and
   interpolate steps, JSON specs, and fitted-pipeline persistence. The API may
   still change before 1.0. The architecture and milestones are in
   `PLAN.md <https://github.com/flex-pse/flex-signal/blob/main/PLAN.md>`_.

Start with the :doc:`tutorials/index`, or go straight to
:doc:`how_to/clean_a_stream`, then :doc:`explanation/signal_frame_and_flags`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
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

   reference/index
   reference/glossary
