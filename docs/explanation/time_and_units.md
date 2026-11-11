# Time and units

## UTC inside, local time only at the edges

Inside flex-signal every timestamp is **UTC**, timezone-aware, and nanosecond resolution. Data from different
sources (a SCADA historian in local time, a lab system in UTC) then lines up without ambiguity, including across
daylight-saving changes.

Timestamps without a time zone (a *naive* index) are taken as **local time on this machine**, and flex-signal
warns that it did so. Most flex-pse work runs on a machine at, or set up for, the plant whose data it reads, so
that is usually right. Say which zone the data is in with `timezone=`, which also silences the warning:

```python
import pandas as pd
import flexsignal as fs

df = pd.DataFrame({"flow": [1.0, 2.0]}, index=pd.date_range("2025-01-15 01:00", periods=2, freq="1h"))
fs.SignalFrame.from_frame(df)    # NaiveTimestampWarning: ... taken as local time on this machine (...)
sf = fs.SignalFrame.from_frame(df, timezone="America/Los_Angeles")
sf.index[0]   # Timestamp('2025-01-15 09:00:00+0000', tz='UTC')
```

`fs.local_timezone()` returns the zone that will be assumed, as an IANA name such as `"America/Los_Angeles"`.
The `TZ` environment variable overrides it on every platform.

```{important}
The same naive data gives different UTC times on machines in different time zones. Pass `timezone=` (or set it
on a spec's `csv` input) whenever a result must be reproducible: in shared specs, in CI, and for data from another
site. `TZ=UTC` makes a whole run independent of the machine. To silence the warning everywhere, use
`warnings.filterwarnings("ignore", category=fs.NaiveTimestampWarning)`.
```

```{note}
Naive local timestamps can't represent daylight-saving changes. When clocks spring forward, an hour of local
times doesn't exist; when they fall back, an hour happens twice. A naive export that spans a change fails to
localize, and flex-signal says so instead of guessing. Export with UTC offsets, or in UTC, when you can.
```

The same goes for CSV files without UTC offsets (`read_csv(..., timezone=...)`) and naive Parquet indexes.
Timestamps written *in a spec*, such as a `window` step's `start`, are different: without an offset they mean
UTC, so a spec means the same thing on every machine.

Local, naive time appears in only one place: the flex-pse boundary. flex-pse models run on the facility's local
wall clock, and the conversion happens there (M3).

## Grids and durations

Durations in specs are pandas offset strings: `"5min"`, `"2h"`, `"1D"`. Steps that need a regular grid (`shift`,
and later the flex-pse adapter) check for one, and tell you to run `align` or `resample` first if the data isn't on
one. Grids from `align` and `resample` are anchored at multiples of the step from the Unix epoch, so the same data
always gives the same grid.

Interpolation gap lengths are measured between the valid samples either side of a gap. On 5-minute data,
`max_gap="15min"` fills up to two missing samples in a row.

## Units are metadata

Each stream's units are a string in its {class}`~flexsignal.core.StreamMeta`, in the pyomo/pint syntax flex-pse
uses (`"m^3/hr"`, with `^`, not `**`). The values stay plain float64, so every library works on them unchanged.

In M1, units are carried and checked, but not converted:

- `combine` refuses streams with different units unless you state the result's units.
- A `range` bound given as `{"value": 50, "units": "m^3/hr"}` must use the stream's own units.

Conversion between units arrives with the `[units]` extra (M5).
