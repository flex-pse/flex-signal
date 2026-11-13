Pipelines and entry points
==========================

Running a spec
--------------

These are the entry points flex-orchestrator will wrap (see PLAN.md §4.3): a
spec goes in and a JSON-able result comes out, with no file I/O when data is
passed in.

.. currentmodule:: flexsignal

.. autofunction:: run

.. autofunction:: fit

.. autoclass:: RunResult
   :members:

Pipelines
---------

.. autoclass:: Pipeline
   :members: from_spec, fit, fit_transform

.. autoclass:: FittedPipeline
   :members: transform, report, to_payload, from_payload, save, load

.. currentmodule:: flexsignal.pipeline

.. autofunction:: bind_inputs

.. autofunction:: step_record
