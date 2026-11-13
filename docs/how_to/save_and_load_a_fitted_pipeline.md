# Save and load a fitted pipeline

Some steps learn from training data. `tukey`, for example, learns each stream's quartiles. You fit once on a
period you trust, save the result, and apply the same fences to new data later, in another process, or on
another machine.

## Fit on training data

```python
import flexsignal as fs
from flexsignal.testing import synthetic

train = synthetic.plant(seed=1).signal()        # a period you trust
pipe = fs.Pipeline(
    [
        fs.clean.Tukey("pressure", k=3),
        fs.impute.Interpolate("pressure", max_gap="15min"),
    ],
    name="pressure_fences",
)
fitted = pipe.fit(train)
print(fitted.states[0])     # {'fences': {'pressure': [lower, upper]}}
```

The fitted state is plain JSON. The spec itself is unchanged by fitting.

## Save it

```python
fitted.save("artifacts/pressure_fences")
```

This writes a directory of JSON files:

```text
artifacts/pressure_fences/
  spec.json                 the pipeline spec
  fitted/manifest.json      format version, spec hash, fit report, step list
  fitted/00_tukey.json      one fitted state per step
  fitted/01_interpolate.json
```

No pickle is involved, so the files are safe to open, diff and version. They also load in any later flex-signal
release that reads the same format version. Saving into a directory that already holds a pipeline needs
`overwrite=True`.

## Load it and apply it to new data

```python
loaded = fs.FittedPipeline.load("artifacts/pressure_fences")
new_week = synthetic.plant(seed=2).with_out_of_range("pressure", count=3, value=50.0).signal()
cleaned = loaded.transform(new_week)
print(cleaned.flag_counts()["pressure"]["flags"])     # {'OUTLIER': 3}
```

`transform` reuses the fences learned from the training week. It doesn't recompute them on the new data.

## Store it somewhere other than a directory

`to_payload()` returns the same content as one dict:

- `spec`;
- `state`;
- `spec_hash`;
- `fit_report`;
- `blobs`, binary state for steps whose state can't be JSON (none yet).

Everything except `blobs` is JSON-able. This is what a database or flex-orchestrator's storage backend keeps:

```python
import json

payload = fitted.to_payload()
text = json.dumps(payload)                       # store this anywhere
restored = fs.FittedPipeline.from_payload(json.loads(text))
assert restored.transform(new_week).equals(cleaned)
```

`from_payload` checks the stored spec against its recorded hash. If they don't match, it refuses to load,
because the fitted state may belong to a different spec.
