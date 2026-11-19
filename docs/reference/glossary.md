# Glossary

```{glossary}
backend
  An adapter, named by a string key such as `sklearn:HuberRegressor` or `pypots:SAITS`, that runs a step's
  algorithm in an existing library. It converts flex-signal's data into the shape that library expects, and back
  again, keeping the time index.

benchmark (imputer)
  A test that hides known values as gaps of a chosen length, lets an imputer fill them, and scores the fill
  against what was hidden. See {func}`flexsignal.eval.benchmark_imputers`.

condition
  A short expression over stream names that gives True or False per row, such as `"C == 0 and B > 0"`. Used by
  `keep_when`, `invalid_when`, `fit_when` and `apply_when`. See [Conditions](../explanation/conditions.md).

gap
  A run of missing values in one stream. Its length is the time between the valid samples either side of it.

input slot
  A named input in a spec, such as `historian`. The caller binds data to it at run time, so a pipeline does no
  I/O of its own.

quality flag
  A per-sample bitmask that records what happened to a value: missing, out of range, outlier, stuck, corrected,
  imputed, forecast, and so on. Cleaning sets a flag instead of deleting data.

reference stream
  A trusted measurement, possibly intermittent, that another stream is calibrated against.

SignalFrame
  flex-signal's data container: a pandas DataFrame of values with a UTC time index, a matching frame of quality
  flags, and per-stream metadata (units, source, provenance).

SignalSpec
  The declarative JSON description of a pipeline: its inputs, steps and outputs, plus a `schema_version`. A spec
  can be stored, versioned and re-run.

step
  One operation in a pipeline, such as `align`, `hampel`, `calibrate`, `interpolate` or `forecast`, identified in
  a spec by its `kind`.

tier
  One stage of a `tiered` imputer: a method for gaps up to a given length.

trusted rows
  In `calibrate`, the rows where the reference has a value and `fit_when` holds. The model is fitted on them, and
  with `prefer_reference` they take the reference's own value.
```
