Core
====

.. automodule:: flexsignal.core
   :no-members:

SignalFrame
-----------

.. currentmodule:: flexsignal.core

.. autoclass:: SignalFrame
   :members:

.. autoclass:: StreamMeta
   :no-members:

.. flexsignal-spec-table:: flexsignal.core.StreamMeta

.. autofunction:: local_timezone

Quality flags
-------------

.. autoclass:: Flag
   :no-members:

.. flexsignal-flag-table::

.. autofunction:: flag_names

.. autofunction:: parse_flags

Conditions
----------

.. automodule:: flexsignal.core.conditions
   :no-members:

.. currentmodule:: flexsignal.core

.. autofunction:: evaluate_condition

.. autofunction:: check_condition

.. autodata:: CONDITION_FUNCTIONS
   :no-value:

Errors
------

.. autoexception:: SignalError

.. autoexception:: SignalConfigError

.. autoexception:: SignalDataError

.. autoexception:: SignalBackendError

.. autoexception:: SignalWarning

.. autoexception:: NaiveTimestampWarning

The step registry
-----------------

.. autofunction:: register_step

.. autofunction:: get_step_class

.. autofunction:: registered_steps
