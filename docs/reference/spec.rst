Specs
=====

.. automodule:: flexsignal.spec
   :no-members:

The committed JSON Schema:
:download:`signal_spec.schema.json <../../src/flexsignal/spec/schemas/signal_spec.schema.json>`.
See :doc:`../explanation/spec_schema`.

Loading, dumping and hashing
----------------------------

.. currentmodule:: flexsignal.spec

.. autofunction:: load_spec

.. autofunction:: dump_spec

.. autofunction:: spec_hash

.. autofunction:: canonical_json

.. autofunction:: migrate

.. autofunction:: resolve_sources

.. autodata:: flexsignal.spec.models.SCHEMA_VERSION

.. autodata:: flexsignal.spec.io.MIGRATIONS
   :no-value:

The spec models
---------------

.. autoclass:: SignalSpec
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.SignalSpec

.. autoclass:: StepSpec
   :members: fit, transform

.. autoclass:: Quantity
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.Quantity

Inputs
^^^^^^

.. autoclass:: FrameInput
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.FrameInput

.. autoclass:: ParquetInput
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.ParquetInput

.. autoclass:: CsvInput
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.CsvInput

Outputs
^^^^^^^

.. autoclass:: ParquetOutput
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.ParquetOutput

.. autoclass:: CsvOutput
   :no-members:

.. flexsignal-spec-table:: flexsignal.spec.CsvOutput

Generating the JSON Schema
--------------------------

.. currentmodule:: flexsignal.spec.schema

.. autofunction:: build_json_schema

.. autofunction:: export_json_schema
