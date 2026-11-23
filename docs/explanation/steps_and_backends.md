# Steps and backends

## The step contract

A step subclasses {class}`~flexsignal.spec.StepSpec` and implements up to two methods:

```{code-block} python
def fit(self, sf: SignalFrame) -> dict:           # learn from training data; {} if nothing to learn
def transform(self, sf: SignalFrame, state: dict) -> SignalFrame:
```

This is scikit-learn's fit/transform split, made functional. The step object holds configuration only, and the
fitted state is an explicit JSON dict. That's what lets a fitted pipeline save as JSON and travel to a Ray
worker, and it keeps a spec unchanged when it's fitted.

The built-in steps come in four groups:

- **select**: `align`, `resample`, `window`, `mask`, `rename`, `combine`, `shift`. These put data on a grid,
  choose what to keep, and reshape. They never judge data quality.
- **clean**: `range`, `rate_of_change`, `stuck`, `hampel`, `tukey`, `expression`. These find bad values, set
  them to NaN, and flag why. Two more reshape a signal: `filter` smooths noise, and `detotalize` turns a running
  total into a rate.
- **correct**: `calibrate`. This corrects a stream against a trusted, intermittent reference
  ([Calibration](calibration.md)).
- **impute**: `interpolate`, `kalman`, `regression`, `knn`, `iterative` and `tiered`. These fill gaps and flag
  them ([Imputation tiers](imputation_tiers.md)).

## Wrap, don't reimplement

flex-signal calls existing libraries rather than rebuilding them. A step names the library object it uses by a
**backend key**, such as `"scipy:PchipInterpolator"`. The key is resolved by {func}`flexsignal.backends.get_backend`
the first time it's used, so an optional library costs nothing until a step needs it. When the library is
missing, the error names the extra that installs it, for example
`` pip install flex-signal[pypots] ``.

`get_backend` resolves registered keys only, never an arbitrary `"package.module:attribute"` string, and a
step's `model` must be one of the keys it lists, because loading a spec must never import code. You can register
keys of your own with {func}`~flexsignal.backends.register_backend` from your code, and use them from
[your own steps](#adding-your-own-step) with `get_backend`. The built-in steps don't accept arbitrary keys: `calibrate` and `regression` take the models they
list or a model registered with {func}`~flexsignal.register_model`, and `interpolate` only its fixed methods. The
simpler route to a method of your own is to register a plain function as a cleaner, imputer, forecaster or model,
and use it by name from Python or a JSON spec ([Your own methods](extending.md)). Only registered methods can run,
so a spec still can't make flex-signal import code.

```{eval-rst}
.. flexsignal-backend-table::
```

The exceptions are checks so small that a dependency would cost more than it saves: range checks, the Hampel
filter, and rate-of-change and stuck-sensor detection. They are a few lines of pandas each, and cite their source.

## Adding your own step

```python
from typing import Literal
from flexsignal.core import Flag, register_step
from flexsignal.steps import StreamStep
from flexsignal.steps._base import remove_values


@register_step
class Negative(StreamStep):
    """Reject negative values."""

    kind: Literal["negative"] = "negative"

    def transform(self, sf, state):
        return remove_values(sf, sf.values[list(self.streams)] < 0, Flag.OUT_OF_RANGE)
```

A step from another package is registered through the `flexsignal.steps` entry-point group, so specs that use
it load wherever the package is installed. For most methods, registering a plain function is simpler: see
[Your own methods](extending.md).
