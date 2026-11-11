# Contributing to flex-signal

Thanks for your interest in contributing! This project welcomes issues, pull requests, and discussion from anyone.
flex-signal is part of the [flex-pse](https://github.com/flex-pse/flex-pse) ecosystem and follows the same
contribution standards, so it stays maintainable and trustworthy for everyone. Please read these guidelines before
you submit anything.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Project Rules](#project-rules)
- [Human Readability Standards](#human-readability-standards)
- [Contributor Responsibility](#contributor-responsibility)
- [Use of AI Tools](#use-of-ai-tools)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Spec schema versioning](#spec-schema-versioning)
- [Review & Verification by Maintainers](#review--verification-by-maintainers)
- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs & Security Issues](#reporting-bugs--security-issues)
- [Licensing](#licensing)

---

## Ways to Contribute

- **Bug reports:** open an issue with clear reproduction steps.
- **Feature requests:** open an issue that describes the problem, not just the solution.
- **Code contributions:** open a pull request (see below).
- **Documentation:** fixes, clarifications, and examples are always welcome.
- **Reviewing open PRs and issues:** thoughtful feedback from the community speeds everything up.

## Getting Started

1. Fork the repository and clone it locally.
2. Create the environment: `conda env create -f environment.yml` (or `pip install -e ".[dev,docs]"`).
3. Install the hooks: `pre-commit install --hook-type pre-commit --hook-type pre-push`.
4. Run `pytest` to confirm a clean baseline.
5. Create a branch for your work: `git checkout -b your-short-branch-name`.

## Project Rules

These rules come from decisions in [PLAN.md](PLAN.md). CI enforces most of them.

- **Wrap, don't reimplement.** If pandas, scipy, scikit-learn, statsmodels or an optional library already does
  something, call it through a backend adapter. Small checks are the exception: a range check, a Hampel filter or
  a stuck-sensor test can live in-house.
- **Import direction.**
  - flex-pse imports flex-signal, never the reverse at import time.
  - Only `flexsignal.integrations.flexpse` may import flex-pse (`flexcore`, `flexops`, `flexparameterize`),
    `pyomo` or `idaes`, and it does so lazily, inside functions.
  - Only `flexsignal.integrations.acquirium` may import `acquirium` or `polars`.
  - Optional backends (torch, PyPOTS, ...) are imported lazily as well.
  - `lint-imports` checks all of this, and a unit test checks that `import flexsignal` stays light.
- **Errors say what to do.** Every error subclasses the flex-signal error hierarchy. Its message says what was
  wrong and what the user should do about it, naming the spec field where there is one.
- **Tests.** Tests live next to the code, in `src/flexsignal/<pkg>/tests/`. Every test carries exactly one tier
  marker:
  - `unit`: under a second, no optional extras;
  - `component`: small end-to-end pipelines;
  - `integration`: larger synthetic datasets, flex-pse or optional extras.

  Algorithm tests are deterministic, with fixed seeds and synthetic data with known injected faults. Every bug fix
  comes with a test that reproduces the bug.
- **No outside data.** Tests, examples, benchmarks and docs use only two kinds of data:
  - data from flex-signal's synthetic generator (`flexsignal.testing.synthetic`);
  - the public flex-pse test dataset vendored in `src/flexsignal/tests/data/flexpse/`, whose README records its
    provenance.

  Do not copy data from other repositories, and do not commit plant or customer data. If a bug only shows up on
  real data, reproduce it with a synthetic case.
- **Tests first.** Write the tests for a new step or backend before the implementation, and confirm they fail for
  the right reason (an `ImportError`, not a typo) before writing the code.
- **Validate methods and cite them.** A cleaning, imputation or forecasting method, and its default parameters
  (a Hampel window, a Tukey fence, a gap-length threshold), should rest on the documentation of the library it
  wraps, peer-reviewed literature, or a documented benchmark. Cite the source in the step's docstring or its
  explanation page. State the method's range of validity: the gap lengths, sampling rates and data properties it
  assumes. Never fabricate a citation or a benchmark result.
- **Docs ship with the code.** A feature is not done until its explanation, how-to and reference pages exist and
  `sphinx-build -W` passes. Public functions, classes and pydantic fields need Google-style docstrings or
  `description`s. Ruff's `D` rules enforce this.
- **Changelog.** Add an entry under `[Unreleased]` in `CHANGELOG.md`.

## Human Readability Standards

Code and documentation in this repository are read far more often than they are written. All contributions must be
**understandable by a human reviewer without special tooling**. Specifically:

- **Clear naming.** Name variables, functions and files for what they do, without cryptic abbreviations.
- **Comments explain "why," not "what."** Don't restate the code in prose. Explain intent, trade-offs, and
  non-obvious decisions.
- **Reasonable diffs.** Keep each pull request to a single logical change. Large, unrelated changes bundled
  together will be asked to be split up.
- **Consistent style.** Follow the formatting and linting configuration in this repo, and run it before
  submitting.
- **Docs and comments must read as if a person wrote them for another person.** Use plain, direct language,
  without padding or filler.

## Contributor Responsibility

By submitting a contribution, **you are personally vouching for it**. Specifically, you confirm that:

- You wrote or meaningfully reviewed the content. You are not simply relaying output you don't understand and
  asking maintainers to check it for you.
- You have tested the change yourself, locally, before opening the PR.
- You have the right to submit the contribution: it's your own work, or it's appropriately licensed and
  attributed.
- You will respond to review feedback on your own submission. A contribution that goes 10 days without a response
  to requested changes may be closed.

Maintainers are volunteers reviewing in good faith. Submitting content you can't explain or defend shifts your
verification burden onto them. That isn't a fair trade, and it will typically result in the PR being closed.

## Use of AI Tools

AI-assisted contributions (code, docs, or issue text) are welcome, but the same responsibility bar applies as above.
The tool doesn't lower it.

- **Disclose it.** If a substantial part of a PR or issue was generated or drafted with an AI tool, say so in the
  PR description. A short note is enough; there's no need to detail every prompt.
- **You must still understand and verify it.** Review, test, and be prepared to explain any AI-assisted content as
  if you wrote it yourself. "The AI generated it" is not an acceptable answer to a review question.
- **No unreviewed bulk submissions.** Large batches of AI-generated issues or PRs that no human has individually
  vetted will be closed without full review.
- **Fabricated content is not acceptable.** Do not submit AI-generated citations, benchmarks, test results, or
  changelog entries that you haven't personally checked against reality.

## Submitting a Pull Request

1. Make sure your branch is up to date with `main`.
2. Write a clear PR description: what changed, why, and how you tested it.
3. Link any related issues.
4. Make sure CI passes (lint, import contracts, tests, and the docs build) before requesting review.
5. Keep the scope of the PR narrow: one concern per PR.
6. If the PR adds or changes a step `kind`, a spec field or a backend, add it to the spec round-trip tests and to
   the generated reference tables, or explain in the PR why not.

## Spec schema versioning

- Any change to `SignalSpec` bumps its `schema_version` and regenerates the committed JSON Schema in the same PR.
- Once a spec version has been released, a breaking change also adds a `MIGRATIONS` hook from the old version. A
  migration that cannot translate a spec raises an error that names the field and how to fix it.

## Review & Verification by Maintainers

All contributions go through maintainer review before merging. This is a verification step, not a rubber stamp:

- **Maintainer approval** is required before merge.
- Maintainers may request changes, ask clarifying questions, or ask the contributor to explain specific parts of
  the change. Please respond rather than resubmitting silently.
- Automated checks (tests, linting, import contracts, docs build) must pass. Passing CI does not guarantee a
  merge: a human still reviews correctness, readability, and intent.
- Maintainers may decline a contribution that is functionally correct but doesn't meet the standards above, if it
  would harm long-term readability or maintainability.

## Code of Conduct

This project follows a Code of Conduct. Please be respectful and constructive in issues, PRs, and discussions.

## Reporting Bugs & Security Issues

- Regular bugs: open a GitHub issue with reproduction steps.
- Security vulnerabilities: do **not** open a public issue. Report them privately through GitHub's
  "Report a vulnerability" button on the repository's Security tab.

## Licensing

By contributing, you agree that your contributions will be licensed under this project's Apache 2.0 license (see
`LICENSE`).
