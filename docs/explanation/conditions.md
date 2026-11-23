# Conditions

A **condition** is a short expression over stream names that gives True or False for each row. Steps use them to
choose rows:

| Field | Step | Meaning |
|---|---|---|
| `keep_when` | `mask` | Keep data only where this holds |
| `invalid_when` | `expression` | Reject values where this holds |
| `fit_when` | `calibrate` | Fit only on rows where this holds |
| `apply_when` | `calibrate` | Correct only where this holds |

```python
import pandas as pd
import flexsignal as fs
from flexsignal.core import evaluate_condition

index = pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC")
sf = fs.SignalFrame.from_frame(
    pd.DataFrame({"C": [1, 0, 0, 0, 0, 1], "B": [5.0, 6, None, 8, 9, 10]}, index=index)
)
print(evaluate_condition(sf, "C == 0 and B > 5").tolist())
# [False, True, False, True, True, False]
```

## Syntax

The syntax is Python's, cut down to what a condition needs:

- stream names, numbers, `True` and `False`. A name that isn't a Python identifier goes in backticks:
  ``"`flow rate` > 5"``;
- arithmetic: `+ - * / // % **`. The exponent after `**` must be a number no larger than 100;
- comparisons: `== != < <= > >=`, which may be chained, as in `0 < flow <= 50`;
- `and`, `or` and `not`, or their short forms `&`, `|` and `~`;
- math functions such as `abs`, `sqrt`, `log`, `exp`, `minimum` and `maximum`. The full list is
  {data}`~flexsignal.core.CONDITION_FUNCTIONS`.

Two rules differ from plain Python:

- **`&` and `|` bind like `and` and `or`**, as in `pandas.eval`. So `flow > 1 & pump_on == 1` means
  `(flow > 1) and (pump_on == 1)`.
- **A comparison with a missing value is always False**, `!=` included. Where B is missing, `B > 0`, `B <= 0` and
  `B != 0` are all False. To ask whether a value is present, use `valid(B)`.

flex-signal also checks types. A condition must give True or False, so `flow + 1` is refused with a hint to compare
it to something. `and`, `or` and `not` need True/False on both sides, so `flow and pump_on` is refused too. All of
this is checked when a spec is loaded, before any data is read.

## Helpers

Four helpers read things a plain expression can't:

| Helper | True where |
|---|---|
| `held(C == 0, "30min")` | the condition has been true, without a break, for at least 30 minutes |
| `valid(B)` | B has a value |
| `is_raw(B)` | B has a value that no step has touched: no flags at all |
| `flagged(A, "OUTLIER", "STUCK")` | A carries any of the named flags |

`held` is how you skip transients. A meter that settles for a few minutes after a valve closes is only trustworthy
once the valve has been closed for a while:

```python
print(evaluate_condition(sf, 'held(C == 0, "20min")').tolist())
# [False, False, False, True, True, False]
```

`held` counts from the first row of each unbroken run in the data it is given. At the very start of the data it
can't see earlier history, so it starts counting from there.

`is_raw` and `flagged` read flags, so they need a `SignalFrame`, not a bare DataFrame. They see the flags as the
earlier steps left them. A condition like `not flagged(A, "OUTLIER")` therefore only means something after the step
that sets `OUTLIER`.

## Formulas

`calibrate` with `model="scipy:curve_fit"` takes a **formula**, such as `"a + b * A + c * A ** 2"`. A formula uses
the same syntax but gives a number per row. Names listed in `initial` are the parameters to fit; every other name
is a stream. The helpers aren't allowed in formulas.

## Why conditions can't run code

A spec is data, and a spec written by someone else must be safe to load. So flex-signal never passes a condition to
`eval`, or to `pandas.eval`. It parses the text with Python's `ast` module, refuses anything outside the syntax
above, and evaluates the parsed tree itself with numpy. Attribute access (`flow.__class__`), indexing, strings
outside the helpers' arguments, keyword arguments and any other function are refused with an error naming the
field. See [the spec schema](spec_schema.md) for what a spec can and can't do.
