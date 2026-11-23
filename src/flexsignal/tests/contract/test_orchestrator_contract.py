"""The orchestrator contract, checked without flex-orchestrator.

flex-orchestrator will wrap ``flexsignal.run``/``fit`` as commands whose params
model embeds a SignalSpec. These tests pin down what that wrapper relies on.
"""

import json
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel, ValidationError

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


# -- the rest of the contract ----------------------------------------------------------


@pytest.mark.unit
def test_a_spec_object_dict_json_text_and_path_give_the_same_result(tmp_path):
    spec = _spec()
    path = tmp_path / "spec.json"
    fs.dump_spec(spec, path)
    data = {"historian": _data()}
    forms = [spec, fs.dump_spec(spec), json.dumps(fs.dump_spec(spec)), path, str(path)]
    results = [fs.run(s, data, write_outputs=False) for s in forms]
    for result in results[1:]:
        assert result.spec_hash == results[0].spec_hash
        assert result.signal.equals(results[0].signal)


@pytest.mark.unit
def test_a_slot_takes_the_orchestrators_dataframe_and_units():
    signal = _data()
    df, units = signal.values, dict(signal.units)  # what orch.get_data returns
    result = fs.run(_spec(), {"historian": (df, units)}, write_outputs=False)
    assert result.units == units
    assert result.data.index.equals(df.index)


@pytest.mark.unit
def test_naive_timestamps_warn_and_aware_ones_do_not():
    df = _data().values
    naive = df.tz_convert("UTC").tz_localize(None)
    with pytest.warns(fs.NaiveTimestampWarning):
        fs.run(_spec(), {"historian": naive}, write_outputs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fs.NaiveTimestampWarning)
        fs.run(_spec(), {"historian": df}, write_outputs=False)


@pytest.mark.unit
def test_handoff_grids_default_to_utc_as_the_orchestrator_does():
    from flexsignal.handoff import GridOptions

    assert GridOptions().timezone == "UTC"


@pytest.mark.unit
def test_runs_are_repeatable_and_results_pickle():
    data = {"historian": _data()}
    first = fs.run(_spec(), data, write_outputs=False)
    second = fs.run(_spec(), data, write_outputs=False)
    assert first.signal.equals(second.signal)
    assert first.report == second.report
    clone = pickle.loads(pickle.dumps(first))  # a Ray task returns it
    assert clone.signal.equals(first.signal) and clone.spec_hash == first.spec_hash


@pytest.mark.unit
def test_concurrent_runs_of_different_specs_dont_interfere():
    # A Ray worker may run several tasks in one process: no global state leaks.
    other = fs.Pipeline(
        [
            fs.clean.Hampel("flow", window="1h"),
            fs.impute.Interpolate("flow", max_gap="1h"),
        ],
        name="other",
        inputs=[{"kind": "frame", "name": "historian"}],
    ).spec
    specs = [_spec(), other] * 4
    data = {"historian": _data()}
    serial = [fs.run(s, data, write_outputs=False).signal for s in specs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = list(
            pool.map(lambda s: fs.run(s, data, write_outputs=False).signal, specs)
        )
    for a, b in zip(serial, threaded, strict=True):
        assert a.equals(b)


@pytest.mark.unit
def test_the_embedded_schema_lists_every_kind_by_its_stable_name():
    params_schema = SignalRunParams.model_json_schema()
    assert "spec" in params_schema["properties"]
    from flexsignal.core.registry import own_steps
    from flexsignal.spec.schema import build_json_schema

    text = json.dumps(build_json_schema())
    for kind in own_steps():
        assert f'"const": "{kind}"' in text, kind


@pytest.mark.component
def test_backtest_takes_a_spec_and_returns_json():
    spec = fs.Pipeline(
        [fs.forecast.SeasonalNaive("flow", horizon="1h", season="1D")],
        name="bt",
        inputs=[{"kind": "frame", "name": "historian"}],
    ).spec
    data = {"historian": synthetic.plant(seed=2, periods=4 * 288).signal()}
    seen = []
    result = fs.backtest(
        fs.dump_spec(spec),
        data,
        folds=2,
        compare=[fs.forecast.Persistence("flow", horizon="1h")],
        progress=lambda f, m: seen.append(f),
    )
    as_json = json.loads(json.dumps(result.to_dict()))
    assert as_json == result.to_dict()
    assert seen and seen[-1] == 1.0


@pytest.mark.unit
def test_a_params_model_checks_and_migrates_the_spec_version(monkeypatch):
    # The params model validates the spec itself, without load_spec: the version
    # must still be checked, and an older one migrated.
    newer = {**fs.dump_spec(_spec()), "schema_version": "9.0.0"}
    with pytest.raises(ValidationError, match="newer than this flex-signal"):
        SignalRunParams(spec=newer, output_dataset="x")

    def rename(data):
        data = dict(data)
        data["name"] = data.pop("title")
        return {**data, "schema_version": fs.SCHEMA_VERSION}

    monkeypatch.setitem(fs.spec.MIGRATIONS, "0.0.9", rename)
    older = {"schema_version": "0.0.9", "title": "old", "steps": []}
    assert SignalRunParams(spec=older, output_dataset="x").spec.name == "old"


@pytest.mark.unit
def test_the_full_json_schema_is_public():
    schema = fs.spec.build_json_schema()
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert "steps" in schema["properties"]
