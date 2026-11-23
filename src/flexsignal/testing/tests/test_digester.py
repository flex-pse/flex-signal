"""synthetic.digester: a two-pool gas response to feed, an offset meter, lab checks."""

import numpy as np
import pytest

from flexsignal.testing import synthetic


@pytest.mark.unit
def test_streams_units_and_seeded_repeatability():
    a, b = synthetic.digester(seed=3), synthetic.digester(seed=3)
    assert list(a.observed.columns) == ["feed", "biogas", "biogas_lab"]
    assert set(a.units.values()) == {"m^3/hr"}
    assert a.observed.equals(b.observed)
    assert not a.observed.equals(synthetic.digester(seed=4).observed)


@pytest.mark.unit
def test_the_meter_reads_the_offset_too_high_and_the_lab_doesnt():
    case = synthetic.digester(seed=1, meter_offset=12.0)
    meter_error = case.observed["biogas"] - case.underlying["biogas"]
    assert meter_error.mean() == pytest.approx(12.0, abs=0.5)
    lab = case.observed["biogas_lab"].dropna()
    assert (lab - case.underlying["biogas"][lab.index]).abs().max() < 3.0
    assert set(lab.index.hour) == {8, 12, 16}
    assert case.fault_mask("calibration", "biogas").all()


@pytest.mark.unit
def test_steady_feed_gives_the_total_yield():
    case = synthetic.digester(seed=2, no_feed_blocks=0, periods=60 * 24)
    gas, feed = case.underlying["biogas"], case.underlying["feed"]
    ratio = gas.iloc[-14 * 24 :].mean() / feed.iloc[-14 * 24 :].mean()
    assert ratio == pytest.approx(20.0, rel=0.05)  # fast_yield + slow_yield


@pytest.mark.unit
def test_without_feed_gas_falls_to_a_slowly_decaying_baseline():
    case = synthetic.digester(seed=0, no_feed_blocks=0, periods=14 * 24)
    feed = case.underlying["feed"].to_numpy().copy()
    feed[7 * 24 : 7 * 24 + 12] = 0.0
    # Re-run the generator's two pools on a feed with a 12-hour outage.
    fast_keep, slow_keep = np.exp(-1 / 2.0), np.exp(-1 / 72.0)
    fast, slow, gas = 12.0 * feed[0], 8.0 * 8.0, []
    for q in feed:
        fast = fast_keep * fast + (1 - fast_keep) * 12.0 * q
        slow = slow_keep * slow + (1 - slow_keep) * 8.0 * q
        gas.append(fast + slow)
    gas = np.array(gas)
    outage = gas[7 * 24 : 7 * 24 + 12]
    assert outage[-1] < 0.6 * gas[7 * 24 - 1]  # the fast part is gone
    assert outage[-1] > 0.25 * gas[7 * 24 - 1]  # the slow baseline is not
    # and the generator's own outages show the same: gas stays above zero
    with_outages = synthetic.digester(seed=0, no_feed_blocks=3)
    off = with_outages.underlying["feed"] == 0
    assert off.any() and (with_outages.underlying["biogas"][off] > 20).all()
