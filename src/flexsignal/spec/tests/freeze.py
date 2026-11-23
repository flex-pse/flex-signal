"""Freeze a released spec schema, so later releases are tested against it.

Run once per release that changes the schema, after bumping ``SCHEMA_VERSION``::

    python -m flexsignal.spec.tests.freeze

It writes ``frozen/<SCHEMA_VERSION>/``:

- ``signal_spec.schema.json``: the JSON Schema as released;
- ``every_step.json``: a spec using every step kind flex-signal provides, with
  inputs, outputs and metadata;
- ``fitted_payload.json``: a fitted pipeline's payload, with the fitted state of
  every core step that learns at fit time;
- ``manifest.json``: the versions and the spec hashes, which must not change.

``test_frozen_specs.py`` then checks that every frozen version still loads
(through ``MIGRATIONS`` when it is older), and that the current version's schema
hasn't changed without a version bump. A frozen directory is never edited.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flexsignal as fs
from flexsignal.spec.schema import schema_text
from flexsignal.testing import synthetic
from flexsignal.testing.examples import one_of_every_step

FROZEN = Path(__file__).parent / "frozen"
"""One directory per released schema version."""


def every_step_spec() -> fs.SignalSpec:
    """Return a spec that uses every step kind and every top-level field."""
    return fs.SignalSpec(
        name="every_step",
        description="One of every step kind, frozen with the schema.",
        requires=["myplant.methods"],
        inputs=[
            {"kind": "frame", "name": "historian"},
            {"kind": "parquet", "name": "lab", "path": "lab.parquet"},
        ],
        steps=one_of_every_step(),
        outputs=[
            {"kind": "parquet", "path": "clean.parquet"},
            {"kind": "csv", "path": "clean.csv", "include_flags": True},
        ],
        metadata={"site": "demo", "owner": "process team"},
    )


def fitted_spec() -> fs.SignalSpec:
    """Return a pipeline of the core steps that store fitted state."""
    return fs.Pipeline(
        [
            fs.clean.Range("flow", min=0, max=500),
            fs.clean.Tukey("pressure", k=3),
            fs.correct.Calibrate(
                target="flow",
                reference="reference_flow",
                model="sklearn:LinearRegression",
                output="flow_cal",
            ),
            fs.impute.Kalman("flow", level="local_level"),
            fs.impute.Regression(
                "pressure", features=["flow"], fourier=[{"period": "1D"}]
            ),
            fs.forecast.Structural(
                "flow_cal", horizon="2h", seasonal=[{"period": "1D", "harmonics": 2}]
            ),
            fs.forecast.Regression(
                "pressure", horizon="1h", lags=["5min"], name="pressure_ahead"
            ),
        ],
        name="fitted",
        inputs=[{"kind": "frame", "name": "historian"}],
    ).spec


def training_data() -> dict[str, Any]:
    """Return the data the frozen payload was fitted on (seeded, synthetic)."""
    case = synthetic.plant(seed=3, periods=3 * 288).with_gaps("flow", count=3)
    return {"historian": case.signal()}


def _write(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def freeze(version: str | None = None) -> Path:
    """Write ``frozen/<version>/``, refusing to overwrite a frozen version."""
    version = version or fs.SCHEMA_VERSION
    target = FROZEN / version
    if target.exists():
        raise SystemExit(
            f"{target} exists: schema {version} is frozen. Bump SCHEMA_VERSION "
            "(and add a migration from it) before freezing again."
        )
    target.mkdir(parents=True)
    (target / "signal_spec.schema.json").write_text(
        schema_text(), encoding="utf-8", newline="\n"
    )
    spec = every_step_spec()
    _write(target / "every_step.json", fs.dump_spec(spec))
    fitted = fs.fit(fitted_spec(), training_data())
    payload = fitted.to_payload()
    _write(target / "fitted_payload.json", payload)
    _write(
        target / "manifest.json",
        {
            "schema_version": version,
            "payload_format_version": payload["format_version"],
            "flexsignal_version": fs.__version__,
            "every_step_hash": fs.spec_hash(spec),
            "fitted_hash": fitted.spec_hash,
        },
    )
    return target


if __name__ == "__main__":
    print(f"froze {freeze()}")
