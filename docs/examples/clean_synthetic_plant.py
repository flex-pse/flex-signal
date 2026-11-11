"""Clean a week of synthetic plant data and report what was removed and filled.

Run it with ``python docs/examples/clean_synthetic_plant.py``. It writes nothing.
"""

import flexsignal as fs
from flexsignal.core import Flag
from flexsignal.testing import synthetic

case = (
    synthetic.plant(seed=7)
    .with_out_of_range("flow", count=4)
    .with_stuck("flow", length=36)
    .with_spikes("flow", count=6, size=40)
    .with_gaps("flow", count=4, length=2)
    .with_out_of_range("pressure", count=3, value=-5.0)
)

pipeline = fs.Pipeline(
    [
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Stuck("flow", min_duration="1h"),
        fs.clean.RateOfChange("flow", max_rate=200, per="1h"),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
        fs.clean.Expression("pressure", invalid_when="pressure < 0"),
        fs.impute.Interpolate(["flow", "pressure"], max_gap="15min"),
    ],
    name="synthetic_plant_cleanup",
)

result = fs.run(pipeline.spec, case.signal())
clean = result.signal

print(f"spec hash: {result.spec_hash[:12]}")
for stream in ("flow", "pressure"):
    print(stream, clean.flag_counts()[stream]["flags"])

removed = clean.has_flag(
    Flag.OUT_OF_RANGE | Flag.STUCK | Flag.RATE_OF_CHANGE | Flag.OUTLIER
)
injected = (
    case.fault_mask("out_of_range", "flow")
    | case.fault_mask("stuck", "flow")
    | case.fault_mask("spike", "flow")
)
print("flow faults removed exactly:", bool(removed["flow"].equals(injected)))
