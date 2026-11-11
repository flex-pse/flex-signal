# Clean a stream

This guide removes sentinel values, a frozen sensor, spikes and short gaps from a flow meter. It also checks the
result against the injected faults. It uses flex-signal's synthetic plant data, so it runs anywhere.

## 1. Get some data

Any DataFrame with a time index works. Timestamps without a time zone are taken as local time on this machine,
with a warning; pass `timezone=` to `SignalFrame.from_frame` to say which zone they are in. Here the synthetic generator plays the part of a historian
export, with four kinds of fault injected:

```python
import flexsignal as fs
from flexsignal.testing import synthetic

case = (
    synthetic.plant(seed=7)                      # 7 days of 5-minute data
    .with_out_of_range("flow", count=4)           # -999 error codes
    .with_stuck("flow", length=36)                # 3 h frozen sensor
    .with_spikes("flow", count=6, size=40)        # isolated spikes
    .with_gaps("flow", count=4, length=2)         # 10-minute dropouts
)
raw = case.signal()          # a SignalFrame; with your own data: fs.SignalFrame.from_frame(df, units=...)
```

## 2. Describe the cleaning

Order matters, because each step sees the data as the steps before it left it:

1. Remove sentinels first, so they can't distort the statistics the later steps compute.
2. Remove stuck runs before the rate check, so the jump at the end of a frozen run isn't mistaken for a spike.
3. Interpolate last, after every bad value has been removed.

```python
pipe = fs.Pipeline(
    [
        fs.clean.Range("flow", min=0, max=500, name="sentinels"),
        fs.clean.Stuck("flow", min_duration="1h"),
        fs.clean.RateOfChange("flow", max_rate=200, per="1h"),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
        fs.impute.Interpolate("flow", max_gap="15min"),
    ],
    name="flow_cleanup",
)
```

How to choose the parameters:

- **`Range`** limits are physical: what the meter can actually read.
- **`RateOfChange`** `max_rate` should sit well above the fastest real change (about 10 m³/h per 5 min here) and
  well below a spike.
- **`Hampel`** needs a window several times wider than a spike but short enough that the daily cycle barely moves
  within it. `min_scale` is the meter's noise level. It stops ordinary noise being flagged on flat stretches,
  where a short window can under-estimate the spread.
- **`Interpolate`** `max_gap` is the longest gap you're willing to bridge with a straight line.

## 3. Run it and look at what happened

```python
result = fs.run(pipe.spec, raw)
clean = result.signal

print(clean.flag_counts()["flow"])
for record in result.provenance:
    print(record["kind"], record.get("samples_changed"))
```

Each provenance record says how many samples a step changed. The flag counts say how many values carry each
reason. Every removed value is still in the frame, as NaN, with its reason in `clean.flags`.

## 4. Check it against the truth

With synthetic data you know exactly where the faults are, so you can check the cleaning precisely:

```python
from flexsignal.core import Flag

removed = clean.has_flag(Flag.OUT_OF_RANGE | Flag.STUCK | Flag.RATE_OF_CHANGE | Flag.OUTLIER)["flow"]
injected = (
    case.fault_mask("out_of_range", "flow")
    | case.fault_mask("stuck", "flow")
    | case.fault_mask("spike", "flow")
)
assert removed.equals(injected), "every injected fault, and nothing else, was removed"

filled = clean.has_flag(Flag.IMPUTED_SHORT)["flow"]
assert filled[case.fault_mask("gap", "flow")].all(), "every short gap was filled"
```

## 5. Keep the spec

The pipeline is data, so save it next to the cleaned results, or hand it to someone else:

```python
import json, pathlib

pathlib.Path("flow_cleanup.json").write_text(json.dumps(fs.dump_spec(pipe.spec), indent=2))
again = fs.run("flow_cleanup.json", raw)
assert again.signal.equals(clean)
```
