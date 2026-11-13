"""The orchestrator contract (PLAN.md §4.3), checked without flex-orchestrator.

flex-orchestrator will wrap ``flexsignal.run``/``fit`` as commands whose params
model embeds a SignalSpec. These tests pin down what that wrapper relies on.
"""

import json
import pickle

import pytest
from pydantic import BaseModel

import flexsignal as fs
from flexsignal.core import SignalError
from flexsignal.testing import synthetic


class SignalRunParams(BaseModel):
    """What an orchestrator command's params model will look like."""

    spec: fs.SignalSpec
    output_dataset: str


def _spec():
    return fs.Pipeline(
        [
            fs.clean.Range("flow", min=0),
            fs.clean.Tukey("pressure", k=3),
            fs.impute.Interpolate("flow", max_gap="15min"),
        ],
        name="contract",
        inputs=[{"kind": "frame", "name": "historian"}],
    ).spec


def _data():
    return synthetic.plant(seed=11, periods=200).with_gaps("flow", count=2).signal()


@pytest.mark.unit
def test_spec_embeds_in_a_params_model_and_round_trips_as_json():
    params = SignalRunParams(spec=_spec(), output_dataset="clean_flow")
    text = params.model_dump_json()
    back = SignalRunParams.model_validate_json(text)
    assert back.spec == params.spec
    assert fs.spec_hash(back.spec) == fs.spec_hash(params.spec)


@pytest.mark.unit
def test_run_takes_named_data_and_returns_orchestrator_shapes():
    result = fs.run(_spec(), {"historian": _data()}, write_outputs=False)
    assert str(result.data.index.tz) == "UTC"
    assert set(result.units) == set(result.data.columns)
    assert result.flags.shape == result.data.shape
    as_json = json.loads(json.dumps(result.to_dict()))
    assert as_json["spec_hash"] == result.spec_hash
    assert json.loads(json.dumps(result.provenance)) == result.provenance


@pytest.mark.unit
def test_run_does_no_io_when_data_is_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fs.run(_spec(), {"historian": _data()})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_payload_is_json_and_fitted_pipelines_pickle():
    fitted = fs.fit(_spec(), {"historian": _data()})
    payload = fitted.to_payload()
    assert payload["blobs"] == {}
    rebuilt = fs.FittedPipeline.from_payload(json.loads(json.dumps(payload)))
    clone = pickle.loads(pickle.dumps(fitted))
    data = {"historian": _data()}
    assert rebuilt.transform(data).equals(fitted.transform(data))
    assert clone.transform(data).equals(fitted.transform(data))
    assert pickle.loads(pickle.dumps(_data())).equals(_data())


@pytest.mark.unit
def test_errors_carry_a_field_path_and_pickle():
    bad = json.loads(json.dumps(fs.dump_spec(_spec())))
    bad["steps"][1]["k"] = -1
    with pytest.raises(SignalError) as exc:
        fs.run(bad, {"historian": _data()})
    assert exc.value.field == "steps.1.k"
    clone = pickle.loads(pickle.dumps(exc.value))
    assert clone.field == "steps.1.k" and str(clone) == str(exc.value)


@pytest.mark.unit
def test_progress_reports_fractions():
    seen = []
    fs.run(_spec(), {"historian": _data()}, progress=lambda f, m: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0
