# Stability and deprecation

flex-signal 0.1.0 is the first release. From it on, a spec or a fitted pipeline that a release saved keeps loading
in every later release. This page says what that promise covers, how changes are made without breaking it, and how
features are deprecated.

## What is frozen

These are stored by other tools, such as flex-orchestrator studies and Acquirium apps, so they are frozen with each
release:

- **The spec schema**: every step `kind`, every option name, type and default, and the top-level fields
  (`inputs`, `steps`, `outputs`, `metadata`, `requires`). Its version is `schema_version`, `0.1.0` today
  ({data}`~flexsignal.spec.models.SCHEMA_VERSION`).
- **Spec hashes.** {func}`~flexsignal.spec.spec_hash` of a given spec stays the same, because downstream tools key
  caches and provenance on it.
- **The fitted-pipeline payload**: {meth}`~flexsignal.FittedPipeline.to_payload` (`format_version` 1), the
  directory layout of {meth}`~flexsignal.FittedPipeline.save`, and the fitted state each step stores.
- **Flags**: the bit of every {class}`~flexsignal.core.Flag`.

What isn't frozen: private modules and names starting with `_`, the wording of error messages and reports, and
numbers that depend on the libraries underneath. A newer scikit-learn or statsmodels may move a fitted
coefficient in its last digits. The adapters (PyPOTS, Chronos, darts and the others) follow their libraries.

## How the freeze is enforced

The test suite holds a frozen copy of each released schema in `src/flexsignal/spec/tests/frozen/<version>/`: the
JSON Schema, a spec that uses every step kind, and a fitted pipeline's payload. The tests check that:

- every frozen spec and payload still loads and runs, migrated if it is older;
- the current schema is the one frozen for the current `schema_version`, so the schema can't change without a new
  version;
- the frozen spec still has the same hash.

## Changing the schema

A change to the schema is a new `schema_version`, following semver:

| Change | Version | Migration |
|---|---|---|
| Add a step kind, or an option with a default | minor (0.1.0 → 0.2.0) | one that only sets the new version |
| Rename or remove an option or kind, change a default or a meaning | minor during 0.x, major from 1.0 | one that rewrites old specs into the new form |

Every change also needs:

1. `SCHEMA_VERSION` bumped in `flexsignal/spec/models.py`;
2. an entry in {data}`~flexsignal.spec.io.MIGRATIONS` that upgrades a spec from the previous version;
3. a new frozen directory, written by `python -m flexsignal.spec.tests.freeze`. Old frozen directories are never
   edited;
4. a CHANGELOG entry that says what changed for spec authors.

A migration takes the spec as a dict and returns it at the next version. {func}`~flexsignal.spec.load_spec` applies
them in turn, so a 0.1.0 spec reaches the current version through every migration in between. See
{doc}`../how_to/migrate_a_spec`.

## Deprecation policy

A feature is deprecated before it is removed, so that code and specs have time to move:

- **Python API.** A deprecated function, argument or class keeps working for at least one minor release and warns
  with a `DeprecationWarning` (a `FutureWarning` where the person who sees it is the end user, not the developer).
  The warning names the replacement and the release that removes it. Removal happens in a later minor release
  during 0.x, and in a major release from 1.0.
- **Specs.** A renamed or removed step kind or option never stops an old spec from loading: its migration rewrites
  it. When there is no faithful rewrite, loading the old spec fails with an error that says what to change.
- **Fitted pipelines.** A new payload `format_version` keeps reading the previous one, or the error says to refit.
  Fitted pipelines are cheap to refit from their spec and training data; specs are what must keep loading.
- **Extras and adapters.** An adapter whose library stops supporting our dependencies (pandas 3, say) is deprecated
  for one minor release and then removed. Its specs fail with an error naming the extra.

Every deprecation and removal is listed in the CHANGELOG under *Deprecated* or *Removed*.

## Before 1.0

0.x releases follow the same rules. The difference is only that a breaking change can come in a minor release, and
always with a migration. 1.0.0 is planned as the stable hand-off to flex-orchestrator.
