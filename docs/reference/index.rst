API reference
=============

The package layout. Each subpackage's purpose and the milestone it is built in
are given below. Lower layers never import higher ones, and only the
integration subpackages touch flex-pse or Acquirium. ``lint-imports`` enforces
both rules.

.. automodule:: flexsignal

Core and spec
-------------

.. automodule:: flexsignal.core

.. automodule:: flexsignal.spec

.. automodule:: flexsignal.io

Steps and pipelines
-------------------

.. automodule:: flexsignal.backends

.. automodule:: flexsignal.steps

.. automodule:: flexsignal.pipeline

.. automodule:: flexsignal.eval

.. automodule:: flexsignal.api

Integrations
------------

.. automodule:: flexsignal.integrations

.. automodule:: flexsignal.integrations.flexpse

.. automodule:: flexsignal.integrations.acquirium
