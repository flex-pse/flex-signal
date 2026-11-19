"""Pipeline edge cases: input binding, fitted-state checks, persistence errors."""

import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError
from flexsignal.pipeline import PAYLOAD_FORMAT, PAYLOAD_VERSION, bind_inputs
from flexsignal.testing import synthetic


def _signal(periods=20):
    return synthetic.plant(periods=periods).signal()


def _pipe(**kwargs):
    return fs.Pipeline([fs.clean.Range("flow", min=0)], name="edges", **kwargs)


# -- binding inputs ------------------------------------------------------------------


@pytest.mark.unit
def test_data_of_the_wrong_type_is_refused():
    with pytest.raises(SignalConfigError, match="got int") as err:
        _pipe().fit(5)
    assert err.value.field == "inputs.data"


@pytest.mark.unit
def test_a_spec_without_inputs_takes_one_named_frame():
    sf = _signal()
    spec = _pipe().spec
    assert bind_inputs(spec, {"anything": sf}) is sf
    assert bind_inputs(spec, (sf.values, {"flow": "m^3/hr"})).units["flow"] == "m^3/hr"
    with pytest.raises(SignalConfigError, match="not 2 named frames"):
        bind_inputs(spec, {"a": sf, "b": sf})


@pytest.mark.unit
def test_several_slots_need_named_frames():
    spec = _pipe(
        inputs=[{"kind": "frame", "name": "scada"}, {"kind": "frame", "name": "lab"}]
    ).spec
    with pytest.raises(SignalConfigError, match=r"2 input slots \['scada', 'lab'\]"):
        bind_inputs(spec, _signal())


@pytest.mark.unit
def test_a_csv_input_is_read_relative_to_base_dir(tmp_path):
    sf = _signal()
    fs.io.write_csv(sf, tmp_path / "raw.csv")
    spec = _pipe(
        inputs=[
            {
                "kind": "csv",
                "name": "raw",
                "path": "raw.csv",
                "timezone": "UTC",
                "units": {"flow": "m^3/hr"},
            }
        ]
    ).spec
    bound = bind_inputs(spec, base_dir=tmp_path)
    assert bound.columns == sf.columns
    assert bound.units["flow"] == "m^3/hr"
    assert bound.values.equals(sf.values)


# -- fitting and transforming --------------------------------------------------------


@pytest.mark.unit
def test_fit_must_return_a_dict():
    class ListState(fs.clean.Range):
        def fit(self, sf):
            return [1, 2]

    with pytest.raises(SignalConfigError, match="returned list from fit") as err:
        fs.Pipeline([ListState("flow", min=0)], name="bad").fit(_signal())
    assert err.value.field == "steps.0"


@pytest.mark.unit
def test_reprs_and_transform_progress():
    pipe = fs.Pipeline(
        [fs.clean.Range("flow", min=0), fs.impute.Interpolate("flow", max_gap="15min")],
        name="edges",
    )
    assert repr(pipe) == "Pipeline('edges': range, interpolate)"
    fitted = pipe.fit(_signal())
    assert repr(fitted) == "FittedPipeline('edges', 2 steps)"
    calls = []
    fitted.transform(_signal(), progress=lambda f, m: calls.append((f, m)))
    assert calls == [(0.5, "transform range"), (1.0, "transform interpolate")]


@pytest.mark.unit
def test_states_must_match_the_steps():
    with pytest.raises(SignalConfigError, match="got 0 fitted states for 1 steps"):
        fs.FittedPipeline(_pipe().spec, [])


# -- persistence ---------------------------------------------------------------------


@pytest.mark.unit
def test_payloads_of_another_format_or_version_are_refused():
    payload = _pipe().fit(_signal()).to_payload()
    with pytest.raises(SignalConfigError, match="not a flex-signal fitted pipeline"):
        fs.FittedPipeline.from_payload({**payload, "format": "other.thing"})
    newer = {**payload, "format_version": PAYLOAD_VERSION + 1}
    with pytest.raises(SignalConfigError, match="unsupported fitted-pipeline"):
        fs.FittedPipeline.from_payload(newer)
    assert payload["format"] == PAYLOAD_FORMAT


@pytest.mark.unit
def test_loading_a_directory_without_a_saved_pipeline(tmp_path):
    with pytest.raises(SignalConfigError, match="doesn't hold a saved pipeline"):
        fs.FittedPipeline.load(tmp_path)
