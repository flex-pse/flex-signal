"""The [anomaly] adapters: PyodDetector, RupturesChangepoint and StumpyDiscord."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyod")
pytest.importorskip("ruptures")
pytest.importorskip("stumpy")

import flexsignal as fs  # noqa: E402
from flexsignal.backends import registered_backends  # noqa: E402
from flexsignal.core import (  # noqa: E402
    Flag,
    SignalBackendError,
    SignalConfigError,
    SignalDataError,
)
from flexsignal.testing import synthetic  # noqa: E402

PLUGIN = "flexsignal.adapters.anomaly"
NAMES = ("PyodDetector", "RupturesChangepoint", "StumpyDiscord")


@pytest.fixture
def anomaly():
    """The anomaly methods, which ``import flexsignal`` registers."""
    return [name for name in NAMES if name in fs.core.registered_steps()]


GLITCHES = ((500, 506, 30.0), (900, 904, -25.0))
"""(start, stop, offset) of transient level glitches added to flow."""


def _case(seed=0, gaps=True):
    """A week of 5-min plant data with spikes, two short glitches and gaps."""
    case = synthetic.plant(seed=seed).with_spikes("flow", count=5, size=40)
    if gaps:
        case = case.with_gaps("flow", count=3, length=2)
    observed = case.observed[["flow", "pressure"]].copy()
    glitch = np.zeros(len(observed), dtype=bool)
    flow = observed["flow"].to_numpy(copy=True)
    for start, stop, offset in GLITCHES:
        flow[start:stop] += offset
        glitch[start:stop] = True
    observed["flow"] = flow
    sf = fs.SignalFrame.from_frame(
        observed, units={"flow": "m^3/hr", "pressure": "bar"}
    )
    spikes = case.fault_mask("spike", "flow").to_numpy()
    return sf, spikes, glitch


def _spec(step):
    return {
        "schema_version": "0.1.0",
        "name": "anomaly",
        "requires": [PLUGIN],
        "steps": [step],
    }


def _hits(signal):
    return signal.has_flag(Flag.OUTLIER)["flow"].to_numpy()


def _check_round_trip(python_step, json_step, sf, tmp_path):
    """Python == JSON, the spec round-trips, a saved fit reloads, repeats agree."""
    from_python = fs.run(fs.Pipeline([python_step], name="anomaly").spec, sf)
    from_json = fs.run(json.dumps(_spec(json_step)), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    spec = from_json.fitted.spec
    assert fs.load_spec(fs.dump_spec(spec)) == spec

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved, plugins=[PLUGIN])
    again = loaded.transform(sf)
    pd.testing.assert_frame_equal(again.values, from_json.signal.values)
    pd.testing.assert_frame_equal(again.flags, from_json.signal.flags)

    repeat = fs.run(json.dumps(_spec(json_step)), sf)
    pd.testing.assert_frame_equal(repeat.data, from_json.data)
    return from_json.signal


# -- registration ---------------------------------------------------------------------


@pytest.mark.unit
def test_methods_and_backend_keys_are_registered(anomaly):
    assert set(NAMES) <= set(anomaly)
    for name in NAMES:
        assert getattr(fs.clean, name).registered.role == "cleaner"
    keys = {b.key: b for b in registered_backends()}
    for key in (
        "pyod:IForest",
        "pyod:ECOD",
        "pyod:COPOD",
        "pyod:HBOS",
        "pyod:KNN",
        "pyod:LOF",
        "ruptures:Pelt",
        "ruptures:Binseg",
        "ruptures:BottomUp",
        "stumpy:stump",
    ):
        assert keys[key].extra == "anomaly"
    fields = fs.clean.PyodDetector.model_fields
    assert "share of present values" in fields["contamination"].description


@pytest.mark.unit
@pytest.mark.parametrize(
    "step, field",
    [
        ({"kind": "PyodDetector", "model": "pyod.models.knn:KNN"}, "model"),
        ({"kind": "PyodDetector", "contamination": "lots"}, "contamination"),
        ({"kind": "PyodDetector", "features": ["value", "nope"]}, "features.1"),
        ({"kind": "RupturesChangepoint", "algorithm": "../Pelt"}, "algorithm"),
        ({"kind": "RupturesChangepoint", "cost": "rbf"}, "cost"),
        ({"kind": "RupturesChangepoint", "min_size": "two"}, "min_size"),
        ({"kind": "StumpyDiscord", "k": "high"}, "k"),
        ({"kind": "StumpyDiscord", "unknown": 1}, "unknown"),
    ],
)
def test_bad_options_fail_when_the_spec_loads(anomaly, step, field):
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(_spec({"streams": ["flow"], **step}))
    assert err.value.field == f"steps.0.{field}"


@pytest.mark.component
@pytest.mark.parametrize(
    "step, error, field",
    [
        (
            {"kind": "PyodDetector", "contamination": 0.9},
            SignalConfigError,
            "contamination",
        ),
        ({"kind": "PyodDetector", "features": []}, SignalConfigError, "features"),
        ({"kind": "PyodDetector", "window": "soon"}, SignalConfigError, "window"),
        (
            {"kind": "RupturesChangepoint", "min_shift": -1.0},
            SignalConfigError,
            "min_shift",
        ),
        (
            {"kind": "RupturesChangepoint", "max_segment": "7min"},
            SignalDataError,
            None,
        ),
        ({"kind": "StumpyDiscord", "window": "10min"}, SignalConfigError, "window"),
        ({"kind": "StumpyDiscord", "window": "3D"}, SignalDataError, None),
    ],
)
def test_bad_option_values_fail_with_a_clear_error(anomaly, step, error, field):
    sf, _, _ = _case()
    with pytest.raises(error) as err:
        fs.run(_spec({"streams": ["flow"], **step}), sf)
    if field is not None:
        assert err.value.field == field
    assert step["kind"] in str(err.value)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["RupturesChangepoint", "StumpyDiscord"])
def test_needs_a_regular_index(anomaly, kind):
    sf, _, _ = _case()
    irregular = sf.values.drop(sf.index[[10, 11, 50]])
    with pytest.raises(SignalDataError, match="regular time index"):
        fs.run(_spec({"kind": kind, "streams": ["flow"]}), irregular)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kind, module",
    [
        ("PyodDetector", "pyod.models.iforest"),
        ("RupturesChangepoint", "ruptures"),
        ("StumpyDiscord", "stumpy"),
    ],
)
def test_a_missing_library_names_the_extra(anomaly, monkeypatch, kind, module):
    monkeypatch.setitem(sys.modules, module, None)
    sf, _, _ = _case()
    with pytest.raises(SignalBackendError, match=r"flex-signal\[anomaly\]") as err:
        fs.run(_spec({"kind": kind, "streams": ["flow"]}), sf)
    assert err.value.value == kind


# -- PyodDetector ---------------------------------------------------------------------


@pytest.mark.component
def test_pyod_runs_from_json_like_python_and_saves(anomaly, tmp_path):
    sf, _, _ = _case()
    python_step = fs.clean.PyodDetector(
        "flow", model="IForest", contamination=0.01, features=["value", "diff"]
    )
    json_step = {
        "kind": "PyodDetector",
        "streams": ["flow"],
        "model": "IForest",
        "contamination": 0.01,
        "features": ["value", "diff"],
    }
    out = _check_round_trip(python_step, json_step, sf, tmp_path)
    assert _hits(out).sum() > 0


@pytest.mark.component
@pytest.mark.parametrize("model", ["IForest", "KNN", "LOF"])
def test_pyod_finds_spikes_and_glitches(anomaly, model):
    sf, spikes, glitch = _case()
    missing = sf.values["flow"].isna().to_numpy()
    out = fs.run(
        fs.Pipeline([fs.clean.PyodDetector("flow", model=model)]).spec, sf
    ).signal
    hits = _hits(out)
    assert hits[spikes].all()
    assert hits[glitch].all()
    # contamination=0.01 rejects about 1% of the present values.
    assert hits.sum() == pytest.approx(0.01 * (~missing).sum(), abs=2)
    # Missing values are skipped, not rejected.
    assert not hits[missing].any()
    assert out.values["pressure"].equals(sf.values["pressure"])


@pytest.mark.component
@pytest.mark.parametrize("model", ["ECOD", "COPOD", "HBOS"])
def test_pyod_other_detectors_run_and_repeat(anomaly, model):
    sf, spikes, _ = _case()
    spec = fs.Pipeline([fs.clean.PyodDetector("flow", model=model)]).spec
    first = fs.run(spec, sf)
    second = fs.run(spec, sf)
    pd.testing.assert_frame_equal(first.flags, second.flags)
    assert 0 < _hits(first.signal).sum() <= 0.01 * len(sf.index) + 2


@pytest.mark.component
def test_pyod_repeats_with_a_fixed_seed(anomaly):
    sf, _, _ = _case()
    a = fs.run(fs.Pipeline([fs.clean.PyodDetector("flow", seed=1)]).spec, sf)
    b = fs.run(fs.Pipeline([fs.clean.PyodDetector("flow", seed=1)]).spec, sf)
    pd.testing.assert_frame_equal(a.flags, b.flags)


@pytest.mark.component
def test_pyod_short_or_empty_streams(anomaly):
    sf, _, _ = _case()
    values = sf.values.iloc[:100].copy()
    values.loc[values.index[20:], "flow"] = np.nan
    with pytest.raises(SignalDataError, match="at least 30"):
        fs.run(_spec({"kind": "PyodDetector", "streams": ["flow"]}), values)
    values["flow"] = np.nan
    out = fs.run(_spec({"kind": "PyodDetector", "streams": ["flow"]}), values)
    assert not _hits(out.signal).any()


# -- RupturesChangepoint --------------------------------------------------------------


@pytest.mark.component
def test_ruptures_runs_from_json_like_python_and_saves(anomaly, tmp_path):
    sf, _, _ = _case()
    python_step = fs.clean.RupturesChangepoint("flow", max_segment="1h", min_shift=6)
    json_step = {
        "kind": "RupturesChangepoint",
        "streams": ["flow"],
        "max_segment": "1h",
        "min_shift": 6,
    }
    _check_round_trip(python_step, json_step, sf, tmp_path)


@pytest.mark.component
def test_ruptures_rejects_short_glitches_and_keeps_regimes(anomaly):
    sf, spikes, glitch = _case()
    values = sf.values.copy()
    flow = values["flow"].to_numpy(copy=True)
    flow[1500:] -= 40.0  # a long regime change: kept
    values["flow"] = flow
    missing = values["flow"].isna().to_numpy()
    out = fs.run(
        fs.Pipeline([fs.clean.RupturesChangepoint("flow")]).spec, values
    ).signal
    hits = _hits(out)
    assert hits[glitch].all()
    assert hits[spikes].all()
    assert not hits[~(glitch | spikes)].any()
    assert not hits[missing].any()


@pytest.mark.component
def test_ruptures_leaves_clean_data_alone(anomaly):
    case = synthetic.plant(seed=3, pump_off_periods=2)
    out = fs.run(
        fs.Pipeline([fs.clean.RupturesChangepoint("flow")]).spec, case.signal()
    ).signal
    assert not _hits(out).any()


@pytest.mark.component
def test_ruptures_max_segment_limits_what_is_rejected(anomaly):
    sf, spikes, glitch = _case()
    out = fs.run(
        fs.Pipeline(
            [fs.clean.RupturesChangepoint("flow", max_segment="20min", cost="l1")]
        ).spec,
        sf,
    ).signal
    hits = _hits(out)
    # The 30-min glitch is now too long to reject; the 20-min one still goes.
    assert not hits[500:506].any()
    assert hits[900:904].all()


# -- StumpyDiscord --------------------------------------------------------------------


@pytest.mark.integration
def test_stumpy_discords(anomaly, tmp_path):
    """One test, so stumpy compiles once per worker (tens of seconds)."""
    sf, spikes, glitch = _case()
    missing = sf.values["flow"].isna().to_numpy()
    python_step = fs.clean.StumpyDiscord("flow", window="1h", k=6.0)
    json_step = {"kind": "StumpyDiscord", "streams": ["flow"], "window": "1h", "k": 6}
    out = _check_round_trip(python_step, json_step, sf, tmp_path)
    hits = _hits(out)
    assert hits[spikes].all()
    assert hits[glitch].all()
    assert not hits[~(glitch | spikes)].any()
    assert not hits[missing].any()

    # A long gap: stretches mostly missing are skipped, nothing near it rejected.
    values = sf.values.copy()
    values.iloc[1200:1260, 0] = np.nan
    hits = _hits(fs.run(fs.Pipeline([python_step]).spec, values).signal)
    assert not hits[1180:1280].any()
    assert hits[spikes & (np.arange(len(hits)) < 1180)].all()

    # Clean periodic data: nothing rejected.
    clean = synthetic.plant(seed=3, pump_off_periods=0).signal()
    assert not _hits(fs.run(fs.Pipeline([python_step]).spec, clean).signal).any()
