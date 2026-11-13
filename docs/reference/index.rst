API reference
=============

The public API. Everything a typical user needs is importable from the top-level
package: ``import flexsignal as fs``, then ``fs.Pipeline``, ``fs.run``,
``fs.clean.Hampel`` and so on.

Lower layers never import higher ones, and only the integration subpackages
touch flex-pse or Acquirium. ``lint-imports`` enforces both rules.

.. toctree::
   :maxdepth: 1

   pipeline
   steps
   spec
   core
   io_backends
   testing
   integrations
