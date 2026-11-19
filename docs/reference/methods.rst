Included methods
================

Every cleaning, imputation and forecasting method that comes with flex-signal.
Each row gives the method's ``fs.`` call (linked to its full reference, with
all its options), the ``kind`` that names it in a JSON spec, the extra that
installs what it needs, and links to the documentation of the library functions
it calls. Methods marked "flex-signal" are implemented here, with nothing but
numpy and pandas.

Every method takes the same ``streams`` argument and runs from Python or from a
JSON spec. Methods of your own are added the same way:
see :doc:`../how_to/add_your_own_method`. The tables are generated from the
code, so they list exactly what this version ships.

Cleaning
--------

Cleaners mark bad values; they are removed, and flagged with why.

.. flexsignal-method-table:: clean

Imputation
----------

Imputers fill gaps and flag each filled value with the method that filled it.
Any of them can be a tier of ``fs.impute.Tiered``.

.. flexsignal-method-table:: impute

Forecasting
-----------

Forecasters write the frame's future rows, with quantiles where asked, and can
be compared in a backtest (:func:`fs.backtest <flexsignal.eval.backtest>`).

.. flexsignal-method-table:: forecast

Correcting a stream against a reference (``fs.correct.Calibrate``) and the
selection steps (resample, align, mask and so on) are in :doc:`steps`.

More models from the same libraries
-----------------------------------

The libraries behind the adapters have many more models than the tables above
list: PyPOTS alone has about 50 imputation models. The tables list the models
flex-signal is **tested** with. The others aren't blocked: name any of a
library's model classes in the adapter's model option, and it runs.

What differs is who sets the model's architecture:

- A **tested** model runs with the adapter's simple settings. For PyPOTS,
  ``n_layers``, ``d_model`` and ``n_heads`` are translated into each model's own
  arguments.
- **Any other** model gets only the settings its constructor shares with the
  tested ones. Anything else it needs goes in ``params``, under the library's
  own argument names. If something required is missing, the error names it and
  links to the library's model list::

      PypotsImpute: PyPOTS PatchTST needs ['patch_size', 'patch_stride', ...],
      which flex-signal doesn't set for untested models: give them in params,
      e.g. params={'patch_size': ...}.

For example, PyPOTS's ImputeFormer, from Python and from JSON:

.. code-block:: python

   fs.impute.PypotsImpute(
       "flow",
       model="ImputeFormer",
       params={"n_layers": 2, "d_input_embed": 32, "d_learnable_embed": 32,
               "d_proj": 16, "d_ffn": 64, "n_temporal_heads": 4},
   )

.. code-block:: json

   {"kind": "PypotsImpute", "streams": ["flow"], "model": "ImputeFormer",
    "params": {"n_layers": 2, "d_input_embed": 32, "d_learnable_embed": 32,
               "d_proj": 16, "d_ffn": 64, "n_temporal_heads": 4}}

A model is always named by its bare class name, never by an import path. It
is looked up among that library's own models, so a spec can't run anything
else. The arguments flex-signal sets itself, such as the window, the horizon,
the number of epochs and the seed, can't be overridden in ``params``: use the
step's own options. Untested models aren't checked by flex-signal's tests, so
compare them in :func:`fs.benchmark_imputers <flexsignal.eval.benchmark_imputers>`
or :func:`fs.backtest <flexsignal.eval.backtest>` before relying on them. Some
models download pretrained weights, e.g. PyPOTS's TimeLLM or darts'
TimesFM3Model. Check that their licence allows your use.

.. flexsignal-model-table::

A library model that doesn't fit an adapter, e.g. one built from other models,
can still be used: write a small function around it and register it with
:func:`fs.register_imputer <flexsignal.register_imputer>` or
:func:`fs.register_forecaster <flexsignal.register_forecaster>`
(:doc:`../how_to/add_your_own_method`). For PyPOTS models,
:func:`~flexsignal.adapters.pypots.to_windows` and
:func:`~flexsignal.adapters.pypots.from_windows` convert between flex-signal's
data and PyPOTS's arrays.
