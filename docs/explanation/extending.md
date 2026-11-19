# Your own methods

flex-signal's built-in steps cover the common cases, but a plant often needs a method of its own: a despiking rule
tuned to one meter, an imputer that knows which pumps share a header, a forecaster a team already trusts. You add
these by **registering** a plain Python function. It then runs like a built-in step: from Python, from a JSON
spec, in a `tiered` imputer or a backtest, and in a fitted pipeline saved as JSON. You don't need to change
flex-signal's code.

## Only registered methods run

A spec is data. It may come from a colleague, from flex-orchestrator, or from a file you downloaded. So one rule
keeps running a spec safe:

> **Only registered methods can run.** A method is registered by code: your script, or a package you installed.
> A spec refers to it by name only. A spec never names an import path, and loading or running a spec never
> imports anything.

A spec from someone else can therefore use only the methods that the person running it chose to make available.
The same rule covers the libraries steps use: a `model` field takes one of the listed keys, never a
`"package.module:Class"` path.

| What a spec can do | What it can't do |
|---|---|
| Use a registered method by name: `{"kind": "PlantDespike", ...}` | Name a function or class to import: `"myplant.methods:despike"` |
| Pick a registered model: `"model": "PlantRidge"` | Make flex-signal import a module, even one listed under `requires` |
| Say which modules it needs: `"requires": ["myplant.methods"]` | Run code through a condition (see [conditions](conditions.md)) |

A spec that names a method nobody registered fails when it loads, before any data is touched, with an error such
as *unknown step kind 'PlantDespike'; register it first, e.g. fs.register_imputer('PlantDespike', fn) ... The
spec says it needs ['myplant.methods']: install them and load them with fs.load_plugins(...)*.

## You write the method; flex-signal does the bookkeeping

| Register with | Your function | flex-signal |
|---|---|---|
| {func}`~flexsignal.register_cleaner` | `fn(values, *, ...)`: given one stream as a Series, returns True for each value to reject | sets them to NaN, adds the flag (`OUTLIER` by default), leaves missing values alone |
| {func}`~flexsignal.register_imputer` | `fn(values, gaps, *, ...)`: given every stream (the one being filled first) and where the gaps are, returns proposed values | finds the gaps (`max_gap`; never `EXCLUDED` or `NO_SAMPLE`), writes only into them, flags them `IMPUTED_MODEL`; usable as a `tiered` tier |
| {func}`~flexsignal.register_forecaster` | `fn(history, future, *, ...)`: given the target and `exog` up to the issue time, and `exog` ahead, returns the forecast's mean (and `std` or quantiles) | the issue time, the forecast rows and their `FORECAST` flag, quantile streams, backtesting |
| {func}`~flexsignal.register_model` | a scikit-learn-style regressor class whose fitted `coef_` and `intercept_` describe it | makes it a `model` for `calibrate` and `regression`; keeps only the coefficients, as JSON |

**Options come from the signature.** The function's keyword-only parameters, after the `*`, become the step's
options. Their type hints and defaults define what is valid, through a generated pydantic model, so a JSON spec
is checked just like a built-in step's, and an error names the field (`steps.3.weight: ...`). A `Literal[...]`
hint is an allowlist. The lines of the function's `Args:` docstring become the options' descriptions. The standard
fields are added by flex-signal: `streams` and `name` always, `max_gap` for imputers, and `horizon`, `issue_time`,
`exog` and `quantiles` for forecasters.

**Fitting is optional.** A `fit=` function learns from the training data and returns a dict. The dict must save
as JSON, as every step's fitted state does, so a fitted pipeline still saves without pickle. The method then
receives it as `state`, after its data arguments. A method without `fit` learns from the data it is given each
time it runs, as the built-in `knn` imputer does.

**Names.** A name must be a Python identifier, and can't be a built-in step's name. It is the step's `kind` in
JSON and its name in `fs.clean`, `fs.impute` or `fs.forecast`. Packages shared between teams should prefix their
names, e.g. `PlantDespike`. Registering the same function again, as a re-run notebook cell does, replaces it;
registering a different function under a taken name is an error.

## Making methods available

The person running a spec chooses which methods exist, in one of three ways:

1. **Register in the script**, before the spec is loaded.
2. **Install a package** that registers its methods through the `flexsignal.steps` entry-point group. They are
   available as soon as the package is installed:

   ```toml
   [project.entry-points."flexsignal.steps"]
   myplant = "myplant.methods"
   ```

3. **Name the modules in the call**: `fs.load_plugins("myplant.methods")`, or `fs.run(spec, data,
   plugins=["myplant.methods"])`. `fs.fit` and `FittedPipeline.load` take `plugins=` too. The list comes from your
   code, never from the spec.

A saved fitted pipeline needs the same methods registered before it loads. For Ray workers, register from an
importable module (option 2 or 3), so every worker can register the same methods; registered steps pickle by name.

## Provenance

Each history record of a registered step names the function behind it: its module, its name, and the version of
the package that provides it. So a result says which code produced it, even when that code isn't flex-signal's.

```{code-block} python
result.provenance[0]["function"]
# {'module': 'myplant.methods', 'function': 'despike', 'package': 'myplant', 'version': '1.2.0'}
```

## flex-signal's own adapters use the same route

The adapters to optional libraries (PyPOTS, Chronos, pyod, darts, sktime, ...) are registered through these same
functions, so the extension API is exercised by flex-signal itself. They are marked "registered" in the
{doc}`step table <../reference/steps>`. Registering one imports nothing heavy: the library is imported when the step
runs, and a missing one raises an error naming the extra that installs it. Writing a full `StepSpec` subclass and
registering it with {func}`~flexsignal.core.register_step` remains the low-level route, for steps that need full
control (see [steps and backends](steps_and_backends.md#adding-your-own-step)).

See [Add your own method](../how_to/add_your_own_method.md) for a worked example.
