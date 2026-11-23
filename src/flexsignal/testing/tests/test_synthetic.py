import numpy as np
import pytest

from flexsignal.testing import synthetic


@pytest.mark.unit
def test_plant_is_deterministic_and_has_expected_streams():
    a, b = synthetic.plant(seed=3), synthetic.plant(seed=3)
    assert a.observed.equals(b.observed)
    assert not a.observed.equals(synthetic.plant(seed=4).observed)
    assert list(a.observed.columns) == list(synthetic.STREAMS)
    assert a.observed.equals(a.clean)
    ref = a.observed["reference_flow"]
    assert ref.isna().any() and ref.notna().any()


@pytest.mark.unit
def test_faults_are_recorded_exactly_and_do_not_overlap():
    case = (
        synthetic.plant(seed=1)
        .with_spikes("flow", count=6)
        .with_stuck("flow", length=36)
        .with_gaps("flow", count=4, length=2)
        .with_out_of_range("flow", count=5)
    )
    assert int(case.fault_mask("spike", "flow").sum()) == 6
    assert int(case.fault_mask("stuck", "flow").sum()) == 36
    assert int(case.fault_mask("gap", "flow").sum()) == 8
    assert int(case.fault_mask("out_of_range", "flow").sum()) == 5
    total = sum(int(case.fault_mask(k, "flow").sum()) for k in case.faults)
    assert int(case.any_fault("flow").sum()) == total
    untouched = ~case.any_fault("flow")
    assert np.array_equal(
        case.observed["flow"][untouched], case.clean["flow"][untouched]
    )
    assert case.fault_mask("spike", "pressure").sum() == 0


@pytest.mark.unit
def test_injection_is_deterministic():
    a = synthetic.plant(seed=2).with_spikes("flow").with_gaps("flow")
    b = synthetic.plant(seed=2).with_spikes("flow").with_gaps("flow")
    assert a.observed.equals(b.observed)


@pytest.mark.unit
def test_drift_and_step_and_errors():
    case = synthetic.plant(seed=0).with_drift("pressure", start=100, rate_per_day=1.0)
    delta = (case.observed["pressure"] - case.clean["pressure"]).iloc[-1]
    assert delta == pytest.approx(
        (case.observed.index[-1] - case.observed.index[100]).total_seconds() / 86400
    )
    stepped = synthetic.plant(seed=0).with_step("flow", start=10, size=5.0)
    offset = (stepped.observed["flow"] - stepped.clean["flow"]).to_numpy()
    assert np.allclose(offset[10:], 5.0) and np.all(offset[:10] == 0)
    with pytest.raises(ValueError, match="no stream"):
        synthetic.plant().with_spikes("nope")
    with pytest.raises(ValueError, match="only room"):
        synthetic.plant(periods=100).with_spikes("flow", count=50)


@pytest.mark.unit
def test_abcd_case():
    case = synthetic.abcd(seed=3, drift_per_day=1.0)
    obs, under = case.observed, case.underlying
    assert list(obs.columns) == ["A", "B", "C", "D"]
    assert case.signal().units == {"A": "m^3/hr", "B": "m^3/hr", "C": None, "D": None}
    reporting = obs["B"].notna()
    assert 0.1 < reporting.mean() < 0.25  # 1 h in every 6 h
    assert set(case.faults) == {
        "calibration",
        "drift",
        "reference_bias",
        "reference_off",
    }
    off = case.fault_mask("reference_off", "B")
    assert (obs["B"][off] == 0).all() and (obs["D"][off] == 1).all()
    biased = case.fault_mask("reference_bias", "B") & ~off
    assert ((obs["B"] - under["B"])[biased] > 15).all()
    good = reporting & ~case.any_fault("B")
    assert ((obs["B"] - under["B"])[good].abs() < 1.5).all()
    # A is miscalibrated everywhere: 0.85 x true + 6 + drift.
    expected = 0.85 * under["A"] + 6.0
    assert ((obs["A"] - expected).iloc[:50].abs() < 4).all()
    assert synthetic.abcd(seed=3).observed.equals(synthetic.abcd(seed=3).observed)
