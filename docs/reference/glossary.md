# Glossary

```{glossary}
backend
  An adapter, named by a string key such as `sklearn:HuberRegressor` or `pypots:SAITS`, that runs a step's
  algorithm in an existing library. It converts flex-signal's data into the shape that library expects, and back
  again, keeping the time index.

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
```
