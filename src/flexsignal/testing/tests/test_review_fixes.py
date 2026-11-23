"""Regression tests for review findings in the synthetic data generators."""

import numpy as np
import pandas as pd
import pytest

from flexsignal.testing import synthetic


def _pump_on_as_first_written(seed, periods, freq, pump_off_periods):
    """The pump-off loop plant() used before it shared _blocks, kept as a reference."""
    rng = np.random.default_rng([seed, 0])
    pump_on = np.ones(periods)
    step_hours = pd.Timedelta(freq).total_seconds() / 3600.0
    for _ in range(pump_off_periods):
        length = int(rng.uniform(2, 6) / step_hours)
        begin = int(rng.integers(0, max(periods - length, 1)))
        pump_on[begin : begin + length] = 0.0
    return pump_on, rng


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 5, 42])
@pytest.mark.parametrize("pump_off_periods", [0, 1, 4, 10])
@pytest.mark.parametrize("freq", ["5min", "15min"])
def test_plant_pump_off_blocks_are_unchanged(seed, pump_off_periods, freq):
    periods = 3 * 288
    case = synthetic.plant(
        seed=seed, periods=periods, freq=freq, pump_off_periods=pump_off_periods
    )
    expected, rng = _pump_on_as_first_written(seed, periods, freq, pump_off_periods)
    assert np.array_equal(case.underlying["pump_on"].to_numpy(), expected)
    # The noise drawn after the blocks must come from the same rng state too.
    sigma = case.noise["flow"]
    noise = rng.normal(0.0, sigma, periods) if sigma else np.zeros(periods)
    flow = case.underlying["flow"].to_numpy()
    assert np.array_equal(case.clean["flow"].to_numpy(), flow + noise)
