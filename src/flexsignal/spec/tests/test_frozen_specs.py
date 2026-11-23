"""Released specs keep loading: the schema is frozen per version.

``frozen/<version>/`` holds what a release wrote (see ``freeze.py``). A spec or a
fitted pipeline saved by any released version must load in every later release,
through ``MIGRATIONS`` if the schema has changed since. And the current version's
schema may not change: a change means a new ``SCHEMA_VERSION``, a migration from
the old one, and a new frozen directory.
"""

import json

import pytest

import flexsignal as fs
from flexsignal.core.registry import own_steps
from flexsignal.spec import MIGRATIONS, SCHEMA_VERSION, dump_spec, load_spec
from flexsignal.spec.schema import schema_text
from flexsignal.spec.tests.freeze import FROZEN, every_step_spec, training_data

VERSIONS = sorted(
    (p.name for p in FROZEN.iterdir() if p.is_dir()),
    key=lambda v: tuple(int(x) for x in v.split(".")),
)

HOW_TO_CHANGE = (
    "Released specs must keep loading. To change the schema: bump SCHEMA_VERSION "
    "in flexsignal/spec/models.py, add a MIGRATIONS entry that upgrades a "
    f"{SCHEMA_VERSION} spec (one that only sets schema_version if the change just "
    "adds something), then run `python -m flexsignal.spec.tests.freeze`. See "
    "docs/explanation/stability.md."
)


def _read(version, name):
    return json.loads((FROZEN / version / name).read_text(encoding="utf-8"))


@pytest.mark.unit
def test_the_current_schema_version_is_frozen():
    assert SCHEMA_VERSION in VERSIONS, (
        f"schema {SCHEMA_VERSION} has no frozen/{SCHEMA_VERSION}/ directory; run "
        "`python -m flexsignal.spec.tests.freeze`."
    )
    assert VERSIONS[-1] == SCHEMA_VERSION, "a frozen version is newer than the code"


@pytest.mark.unit
def test_the_schema_has_not_changed_since_it_was_frozen():
    frozen = (FROZEN / SCHEMA_VERSION / "signal_spec.schema.json").read_text(
        encoding="utf-8"
    )
    assert schema_text() == frozen, f"the spec schema changed. {HOW_TO_CHANGE}"


@pytest.mark.unit
def test_every_step_kind_is_in_the_frozen_spec():
    frozen = {s["kind"] for s in _read(SCHEMA_VERSION, "every_step.json")["steps"]}
    missing = sorted(set(own_steps()) - frozen)
    assert not missing, f"step kinds {missing} aren't frozen. {HOW_TO_CHANGE}"


@pytest.mark.unit
@pytest.mark.parametrize("version", VERSIONS)
def test_every_released_spec_still_loads(version):
    data = _read(version, "every_step.json")
    assert data["schema_version"] == version
    spec = load_spec(data)
    assert spec.schema_version == SCHEMA_VERSION
    assert len(spec.steps) == len(data["steps"])
    if version != SCHEMA_VERSION:
        assert version in MIGRATIONS, f"no migration from schema {version}"


@pytest.mark.unit
def test_the_current_frozen_spec_is_unchanged_and_hashes_the_same():
    data = _read(SCHEMA_VERSION, "every_step.json")
    manifest = _read(SCHEMA_VERSION, "manifest.json")
    spec = load_spec(data)
    assert json.loads(json.dumps(dump_spec(spec))) == data
    # Hashes key caches and provenance downstream, so they must stay stable too.
    assert fs.spec_hash(spec) == manifest["every_step_hash"]
    assert fs.spec_hash(every_step_spec()) == manifest["every_step_hash"], (
        "the examples behind the frozen spec changed; frozen files are never "
        "regenerated, so change testing/examples.py only with a new version."
    )


@pytest.mark.component
@pytest.mark.parametrize("version", VERSIONS)
def test_every_released_fitted_pipeline_still_loads_and_runs(version):
    payload = _read(version, "fitted_payload.json")
    fitted = fs.FittedPipeline.from_payload(payload)
    if version == SCHEMA_VERSION:
        assert fitted.spec_hash == _read(version, "manifest.json")["fitted_hash"]
    out = fitted.transform(training_data())
    ahead = out.has_flag(fs.Flag.FORECAST)
    assert ahead["flow_cal"].sum() == 24 and ahead["pressure"].sum() == 12
    assert not out.values["flow_cal"][ahead["flow_cal"]].isna().any()
    history = out.values.index <= training_data()["historian"].values.index[-1]
    assert not out.values["flow"][history].isna().any()  # Kalman filled the gaps
