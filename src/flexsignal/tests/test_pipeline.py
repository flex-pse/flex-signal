"""M1 acceptance: Python and JSON pipelines agree bit for bit; fitted ones persist."""

import json

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError
from flexsignal.testing import synthetic


def faulty_case(seed=7):
    return (
        synthetic.plant(seed=seed)
        .with_out_of_range("flow", count=4)
        .with_spikes("flow", count=6, size=40)
        .with_stuck("flow", length=36)
        .with_gaps("flow", count=4, length=2)
        .with_out_of_range("pressure", count=3, value=-5.0)
    )


def cleaning_pipeline():
    return fs.Pipeline(
        [
            fs.clean.Range("flow", min=0, max=500, name="sentinels"),
            fs.clean.Stuck("flow", min_duration="1h", exclude_values=[0.0]),
            fs.clean.RateOfChange("flow", max_rate=200, per="1h"),
            fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
            fs.clean.Tukey("pressure", k=3),
            fs.clean.Expression("pressure", invalid_when="pressure < 0"),
            fs.impute.Interpolate("flow", max_gap="15min"),
        ],
        name="plant_cleanup",
        inputs=[{"kind": "frame", "name": "historian"}],
    )


@pytest.mark.component
def test_python_and_json_pipelines_give_bit_identical_output():
    sf = faulty_case().signal()
    pipe = cleaning_pipeline()
    from_python = pipe.fit(sf).transform(sf)
    text = json.dumps(fs.dump_spec(pipe.spec))
    from_json = fs.Pipeline.from_spec(text).fit(sf).transform(sf)
    assert from_python.equals(from_json)
    assert from_python.values.equals(from_json.values)
    assert from_python.flags.equals(from_json.flags)


@pytest.mark.component
def test_full_cleanup_catches_every_injected_flow_fault():
    case = faulty_case()
    result = fs.run(cleaning_pipeline().spec, {"historian": case.signal()}).signal
    removed = (
        result.flags["flow"]
        & int(Flag.OUT_OF_RANGE | Flag.STUCK | Flag.RATE_OF_CHANGE | Flag.OUTLIER)
    ) != 0
    injected = (
        case.fault_mask("out_of_range", "flow")
        | case.fault_mask("spike", "flow")
        | case.fault_mask("stuck", "flow")
    )
    assert np.array_equal(removed.to_numpy(), injected.to_numpy())
    imputed = (result.flags["flow"] & int(Flag.IMPUTED_SHORT)) != 0
    assert imputed[case.fault_mask("gap", "flow")].all()


@pytest.mark.component
def test_history_records_each_step():
    result = fs.run(cleaning_pipeline().spec, faulty_case().signal())
    kinds = [r["kind"] for r in result.provenance]
    assert kinds == [s.kind for s in cleaning_pipeline().spec.steps]
    first = result.provenance[0]
    assert first["name"] == "sentinels" and first["samples_changed"] == {"flow": 4}
    assert first["spec_hash"] == result.spec_hash


@pytest.mark.component
def test_fitted_state_is_reused_on_new_data():
    train = faulty_case(seed=7).signal()
    fitted = cleaning_pipeline().fit(train)
    tukey_index = [s.kind for s in fitted.spec.steps].index("tukey")
    fences = fitted.states[tukey_index]["fences"]["pressure"]
    other = faulty_case(seed=8).signal()
    out = fitted.transform(other)
    pressure = other.values["pressure"]
    expected = (pressure < fences[0]) | (pressure > fences[1])
    flagged = (out.flags["pressure"] & int(Flag.OUTLIER)) != 0
    assert np.array_equal(flagged.to_numpy(), expected.to_numpy())


@pytest.mark.component
def test_save_load_and_payload_round_trip(tmp_path):
    sf = faulty_case().signal()
    fitted = cleaning_pipeline().fit(sf)
    fitted.save(tmp_path / "saved")
    loaded = fs.FittedPipeline.load(tmp_path / "saved")
    assert loaded.spec == fitted.spec and loaded.states == fitted.states
    assert loaded.transform(sf).equals(fitted.transform(sf))
    payload = json.loads(json.dumps(fitted.to_payload()))
    assert (
        fs.FittedPipeline.from_payload(payload)
        .transform(sf)
        .equals(fitted.transform(sf))
    )
    with pytest.raises(SignalConfigError, match="overwrite"):
        fitted.save(tmp_path / "saved")
    fitted.save(tmp_path / "saved", overwrite=True)
    tampered = {**fitted.to_payload(), "spec_hash": "0" * 64}
    with pytest.raises(SignalConfigError, match="spec_hash"):
        fs.FittedPipeline.from_payload(tampered)


@pytest.mark.unit
def test_input_binding_rules(tmp_path):
    sf = synthetic.plant(periods=20).signal()
    spec = cleaning_pipeline().spec
    with pytest.raises(SignalConfigError, match="no data bound"):
        fs.run(spec)
    with pytest.raises(SignalConfigError, match="unknown input slot"):
        fs.run(spec, {"historian": sf, "extra": sf})
    no_inputs = fs.Pipeline([fs.clean.Range("flow", min=0)], name="x").spec
    assert len(fs.run(no_inputs, sf.values).signal) == 20
    with pytest.raises(SignalConfigError, match="no inputs"):
        fs.run(no_inputs)


@pytest.mark.unit
def test_two_inputs_are_joined_and_collisions_rejected():
    flow = synthetic.plant(periods=10).signal().select(["flow"])
    lab = pd.DataFrame(
        {"lab": [1.0, 2.0]},
        index=pd.to_datetime(["2025-01-06 00:07", "2025-01-06 00:20"], utc=True),
    )
    spec = fs.Pipeline(
        [fs.select.Align(freq="5min")],
        name="join",
        inputs=[{"kind": "frame", "name": "scada"}, {"kind": "frame", "name": "lab"}],
    ).spec
    out = fs.run(spec, {"scada": flow, "lab": (lab, {"lab": "mg/L"})}).signal
    assert out.columns == ["flow", "lab"] and out.units["lab"] == "mg/L"
    with pytest.raises(fs.SignalDataError, match="appears in inputs"):
        fs.run(spec, {"scada": flow, "lab": flow})


@pytest.mark.unit
def test_joined_rows_are_not_samples():
    # Regression: the join used to give each stream NaN rows at the other input's
    # times, and align, resample and interpolate treated those as real samples.
    # An off-grid source then aligned to nothing but NaN.
    times = pd.date_range("2025-01-06", periods=12, freq="5min", tz="UTC")
    scada = pd.DataFrame({"flow": np.arange(12.0)}, index=times)
    meter = pd.DataFrame(
        {"meter": np.arange(12.0) * 10}, index=times + pd.Timedelta("2min")
    )
    inputs = [{"kind": "frame", "name": "scada"}, {"kind": "frame", "name": "meter"}]

    joined = fs.run(
        fs.Pipeline([], name="j", inputs=inputs).spec, {"scada": scada, "meter": meter}
    )
    padding = joined.signal.has_flag(Flag.NO_SAMPLE)
    assert int(padding["meter"].sum()) == 12 and int(padding["flow"].sum()) == 12
    assert joined.signal.has_flag(Flag.MISSING)["meter"].equals(padding["meter"])

    def run(*steps):
        spec = fs.Pipeline(list(steps), name="j", inputs=inputs).spec
        return fs.run(spec, {"scada": scada, "meter": meter}).signal

    # The outer grid runs to 01:00, the next grid point after the meter's last
    # sample (00:57); flow's 00:55 sample is the nearest there.
    aligned = run(fs.select.Align(freq="5min"))
    assert aligned.values["meter"].iloc[:12].tolist() == (np.arange(12.0) * 10).tolist()
    assert not aligned.has_flag(Flag.NO_SAMPLE).to_numpy().any()
    assert aligned.has_flag(Flag.RESAMPLED)["meter"].all()
    assert aligned.has_flag(Flag.RESAMPLED)["flow"].sum() == 1

    hourly = run(fs.select.Resample(freq="1h"))
    assert hourly.values["meter"].item() == pytest.approx(55.0)
    assert hourly.flags["meter"].item() == int(Flag.RESAMPLED)

    filled = run(fs.impute.Interpolate("meter", max_gap="15min"))
    assert not filled.has_flag(Flag.IMPUTED_SHORT)["meter"].any()
    assert filled.values["meter"].notna().sum() == 12


@pytest.mark.unit
def test_file_inputs_and_outputs(tmp_path):
    sf = synthetic.plant(periods=50).signal()
    fs.io.write_parquet(sf, tmp_path / "raw.parquet")
    spec = {
        "schema_version": fs.SCHEMA_VERSION,
        "name": "files",
        "inputs": [{"kind": "parquet", "name": "raw", "path": "raw.parquet"}],
        "steps": [{"kind": "range", "streams": ["flow"], "min": 0}],
        "outputs": [
            {"kind": "parquet", "path": "out/clean.parquet"},
            {"kind": "csv", "path": "out/clean.csv", "include_flags": True},
        ],
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    result = fs.run(tmp_path / "spec.json")
    assert len(result.outputs) == 2
    assert fs.io.read_parquet(tmp_path / "out" / "clean.parquet").equals(result.signal)
    dry = fs.run(tmp_path / "spec.json", write_outputs=False)
    assert dry.outputs == ()


@pytest.mark.unit
def test_progress_callback_and_non_json_state():
    calls = []
    fs.run(
        cleaning_pipeline().spec,
        faulty_case().signal(),
        progress=lambda f, m: calls.append((f, m)),
    )
    assert calls[-1][0] == 1.0 and len(calls) == len(cleaning_pipeline().spec.steps)

    class BadState(fs.clean.Range):
        def fit(self, sf):
            return {"array": np.arange(3)}

    with pytest.raises(SignalConfigError, match="isn't JSON-able"):
        fs.Pipeline([BadState("flow", min=0)], name="bad").fit(
            synthetic.plant(periods=5).signal()
        )
