# SignalFrame and quality flags

## One container, plain pandas inside

Every step takes and returns a {class}`~flexsignal.core.SignalFrame`. It isn't a new timeseries library. It's
four plain parts:

| Part | What it is |
|---|---|
| `values` | A pandas DataFrame, one float64 column per stream, indexed by time |
| `flags` | A DataFrame of the same shape: a `uint16` quality bitmask per sample |
| `meta` | Per-stream {class}`~flexsignal.core.StreamMeta`: units, quantity kind, source, description |
| `history` | One provenance record per step that produced the frame |

`sf.values` and `sf.flags` give you ordinary DataFrames, so any pandas, scipy or scikit-learn call works on them
directly. The frame itself is immutable: every step returns a new one.

The time index is always **timezone-aware UTC**, sorted, unique, and nanosecond resolution. A naive index is
taken as this machine's local time, with a warning, unless you say which time zone it is in (see
[Time and units](time_and_units.md)).

```python
import flexsignal as fs
from flexsignal.testing import synthetic

sf = synthetic.plant(seed=0).with_gaps("flow", count=2).signal()
sf.values.head()          # a plain DataFrame
sf.units["flow"]          # 'm^3/hr'
sf.flag_counts()["flow"]  # {'rows': 2016, 'valid': 2012, 'flags': {'MISSING': 4}}
```

## Cleaning never deletes data

A cleaner doesn't drop rows or overwrite values with guesses. When it rejects a value, it does two things:

1. sets the value to NaN;
2. sets a flag bit saying why, for example `OUTLIER` or `STUCK`.

An imputer later fills the NaN, *keeps* the bits that explain why it was missing, and adds `IMPUTED_SHORT`. A
downstream consumer can always tell:

- which values are raw measurements (flags `0`);
- which were removed, and by which rule;
- which were filled in, and how.

```python
from flexsignal.core import Flag, flag_names

flag_names(int(Flag.OUTLIER | Flag.IMPUTED_SHORT))   # ['OUTLIER', 'IMPUTED_SHORT']
imputed = sf.has_flag(Flag.IMPUTED_SHORT)            # boolean DataFrame
```

## The flags

```{eval-rst}
.. flexsignal-flag-table::
```

`EXCLUDED` is deliberately different from the quality flags. It marks data a selection rule left out, such as
"only while the pump runs". That isn't bad data, and imputers never fill it.

## Provenance

When a {class}`~flexsignal.Pipeline` runs, it compares each step's input with its output and appends a record to
the frame's `history`:

- the step's index, kind and name;
- the spec hash;
- the row counts before and after;
- streams added and removed;
- how many samples changed in each stream.

Steps don't write these records themselves, so the record is always exact.
