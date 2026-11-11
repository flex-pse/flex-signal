# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Project scaffold (M0).**
  - `flex-signal` distribution with the `flexsignal` import package: a src layout, Python ≥3.11, and the core
    dependencies from `PLAN.md` §5.1.
  - Subpackage skeleton (`core`, `spec`, `io`, `backends`, `steps`, `pipeline`, `eval`, `api`,
    `integrations.flexpse`, `integrations.acquirium`), with each module's purpose and target milestone in its
    docstring.
  - import-linter contracts for the layering, the independence of the integrations, and the rules that only
    `integrations.flexpse` may import flex-pse, pyomo or idaes and only `integrations.acquirium` may import
    acquirium or polars.
  - Tooling: ruff (flex-pse's rules plus `D`, Google style), black, pre-commit with a pre-push suite, and pytest
    tier markers (`unit`, `component`, `integration`). A root `conftest.py` fails collection when a test has
    anything other than exactly one tier marker.
  - CI mirroring flex-pse v0.3.0:
    - uv installs;
    - tests on Python 3.11–3.14 (Linux), plus 3.13 on macOS and Windows;
    - a single coverage run uploaded to Codecov (`codecov.yml`);
    - an integration job;
    - a standard-install check against the built wheel.
  - Sphinx docs (furo, MyST) deployed to GitHub Pages, and a PyPI publish workflow.
  - `CONTRIBUTING.md`, plus issue and PR templates mirroring flex-pse.
- **Test:** `import flexsignal` loads no flex-pse, pyomo, idaes, torch, polars or acquirium (`PLAN.md` D1).
