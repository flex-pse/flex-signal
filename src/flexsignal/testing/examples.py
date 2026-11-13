"""Ready-made example steps, for tests and docs.

The spec tests check that :func:`one_of_every_step` covers every registered step
kind, so adding a step kind means adding an example here.
"""

from __future__ import annotations

from flexsignal.spec import Quantity, StepSpec
from flexsignal.steps import clean, impute, select

__all__ = ["one_of_every_step"]


def one_of_every_step() -> list[StepSpec]:
    """Return one configured instance of every built-in step kind.

    The options are chosen to exercise non-default values, so a round trip
    through JSON tests more than the defaults.
    """
    return [
        select.Align(freq="5min", method="linear", how="inner"),
        select.Resample(freq="15min", agg="median", min_count=2, name="to_15min"),
        select.Window(start="2025-01-06", end="2025-01-08T12:00:00Z"),
        select.Mask(keep_when="pump_on == 1", streams=["flow"]),
        select.Rename(mapping={"temperature": "temp"}),
        select.Combine(output="flow_best", streams=["reference_flow", "flow"]),
        select.Shift("pressure", by="-15min"),
        clean.Range("flow", min=0.0, max=Quantity(value=500, units="m^3/hr")),
        clean.RateOfChange("flow", max_rate=200, per="1h"),
        clean.Stuck("flow", min_duration="1h", tolerance=0.0, exclude_values=[0.0]),
        clean.Hampel(["flow", "pressure"], window="1h", n_sigma=6),
        clean.Tukey("pressure", k=3),
        clean.Expression("pressure", invalid_when="pressure < 0", flag="OUT_OF_RANGE"),
        impute.Interpolate("flow", max_gap="15min", method="pchip"),
    ]
