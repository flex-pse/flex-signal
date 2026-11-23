import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pickle

    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" are not flex-signal; in your own
    # script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from pydantic import BaseModel, ValidationError

    import flexsignal as fs
    from flexsignal.core import Flag
    from flexsignal.testing import synthetic

    return (
        BaseModel,
        Flag,
        ValidationError,
        fs,
        json,
        mo,
        pd,
        pickle,
        plt,
        synthetic,
    )


@app.cell
def _(plt):
    # One palette and axis style for every tutorial, so the plots read alike.
    COLORS = {
        "signal": "#2b6cb0",
        "truth": "#a0aec0",
        "fault": "#dd6b20",
        "filled": "#2f855a",
        "excluded": "#edf2f7",
    }

    def style(ax, ylabel=None):
        """Light grid, no box, muted ticks."""
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#a0aec0")
        ax.tick_params(colors="#4a5568", labelsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, color="#4a5568")
        return ax

    plt.rcParams["figure.dpi"] = 110
    return COLORS, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 11. A cleaning spec as a flex-orchestrator study

    flex-orchestrator runs studies: it loads data, runs commands on it, stores
    the results and keeps a ledger of what ran. It will run flex-signal
    pipelines through a thin wrapper, so a cleaning spec becomes one step of a
    study. flex-signal ships no orchestrator code. Instead its API keeps a
    contract that makes the wrapper short.

    This tutorial walks through that contract in plain Python, without
    flex-orchestrator:

    1. a spec with a named input slot, and its hash;
    2. a command's params model that embeds the spec;
    3. running the spec on data the orchestrator passes in;
    4. storing a fitted pipeline and using it later, or on another worker;
    5. errors and progress a ledger can record;
    6. spec versions, so stored studies keep loading.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The spec, with a named input

    The spec cleans `flow` and `pressure` and fills short gaps. Its input is a
    `frame` slot called `historian`. A frame slot reads nothing: the caller
    fills it at run time. In a study, that's the data the orchestrator has
    already loaded, so the pipeline itself does no I/O.
    """)
    return


@app.cell
def _(fs, json, mo):
    pipe = fs.Pipeline(
        [
            fs.clean.Range("flow", min=0, max=500),
            fs.clean.Tukey("pressure", k=3),
            fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
            fs.impute.Interpolate(["flow", "pressure"], max_gap="15min"),
        ],
        name="historian_cleanup",
        inputs=[{"kind": "frame", "name": "historian"}],
    )
    spec = pipe.spec
    spec_json = json.dumps(fs.dump_spec(spec), indent=2)
    # Display only (marimo): render the result as formatted text.
    mo.md(f"```json\n{spec_json}\n```")
    return spec, spec_json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `fs.spec_hash` is a SHA-256 of the spec's canonical JSON. It doesn't
    depend on key order or on whether defaults are written out, so it works as
    a cache key ("has this recipe already run on this data?") and as a
    provenance stamp on every stored result.
    """)
    return


@app.cell
def _(fs, json, mo, spec, spec_json):
    key = fs.spec_hash(spec)
    _reordered = dict(reversed(list(json.loads(spec_json).items())))
    same_hash_any_order = fs.spec_hash(fs.load_spec(_reordered)) == key
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"`spec_hash` → `{key}`\n\n"
        f"Same hash with the keys in reverse order: **{same_hash_any_order}**"
    )
    return key, same_hash_any_order


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A command's params model embeds the spec

    An orchestrator command takes a pydantic params model, which the study
    config fills from JSON or YAML. `fs.SignalSpec` is itself a pydantic model,
    so it can be a field. Here the command also takes the name of the dataset
    to write the result to.

    Validating the params checks the spec as `fs.load_spec` does: its
    `schema_version` is checked and an older spec is upgraded (see the last
    section), and every step's options are validated.
    """)
    return


@app.cell
def _(BaseModel, fs):
    class CleanSignalsParams(BaseModel):
        """Params of an orchestrator command that runs a cleaning spec."""

        spec: fs.SignalSpec
        output_dataset: str

    return (CleanSignalsParams,)


@app.cell
def _(CleanSignalsParams, fs, json, key, mo, spec):
    study_json = json.dumps(
        {"spec": fs.dump_spec(spec), "output_dataset": "historian_clean"}
    )
    params = CleanSignalsParams.model_validate_json(study_json)
    params_round_trip = (
        params.spec == spec
        and fs.spec_hash(params.spec) == key
        and CleanSignalsParams.model_validate_json(params.model_dump_json()) == params
    )
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"Validated from JSON: dataset `{params.output_dataset}`, "
        f"{len(params.spec.steps)} steps. Same spec and hash after a JSON "
        f"round-trip: **{params_round_trip}**"
    )
    return params, params_round_trip


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The JSON Schema

    A study config's schema can include the spec's. `model_json_schema()` on
    the params model gives the outline of a spec (name, inputs, outputs and a
    list of steps), but each step there is only `kind` and `name`, because new
    step kinds can be registered at run time. The schema with every built-in
    step's options comes from `fs.spec.build_json_schema()`
    (the same schema committed as `signal_spec.schema.json`). Put it in place
    of the outline and an editor can check each step's options:
    """)
    return


@app.cell
def _(CleanSignalsParams, fs, pd):
    outline = CleanSignalsParams.model_json_schema()
    outline_step_fields = sorted(outline["$defs"]["StepSpec"]["properties"])

    full_spec = fs.spec.build_json_schema()
    full_spec.pop("$schema")
    study_schema = {
        **outline,
        "$defs": {
            **outline["$defs"],
            **full_spec.pop("$defs"),
            "SignalSpec": full_spec,
        },
    }
    step_kinds = sorted(
        study_schema["$defs"]["SignalSpec"]["properties"]["steps"]["items"][
            "discriminator"
        ]["mapping"]
    )
    tukey_k = study_schema["$defs"]["Tukey"]["properties"]["k"]
    schema_table = pd.DataFrame(
        [
            {
                "schema": "model_json_schema()",
                "fields of a step": ", ".join(outline_step_fields),
            },
            {
                "schema": "with fs.spec.build_json_schema()",
                "fields of a step": f"{len(step_kinds)} kinds, each with its "
                f"options, e.g. tukey's k: {tukey_k}",
            },
        ]
    )
    schema_table
    return outline_step_fields, schema_table, step_kinds, study_schema, tukey_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Running the spec on the orchestrator's data

    The orchestrator binds data to the slot by name. `write_outputs=False`
    keeps the run free of file I/O even if the spec lists outputs; the
    orchestrator stores the result itself.

    The data is two days of the synthetic plant, with spikes and short gaps
    put into `flow` at known places.
    """)
    return


@app.cell
def _(synthetic):
    case = (
        synthetic.plant(seed=3, periods=2 * 288)
        .with_spikes("flow", count=6, size=40)
        .with_gaps("flow", count=3, length=2)
    )
    raw = case.signal().select(["flow", "pressure"])
    return case, raw


@app.cell
def _(fs, params, raw):
    result = fs.run(params.spec, {"historian": raw}, write_outputs=False)
    result.data.head()
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `RunResult` has the parts an orchestrator stores, in the shapes
    `orch.store_data(name, df, units=)` takes:

    | attribute | what it is |
    |---|---|
    | `.data` | values, a DataFrame with a tz-aware UTC index |
    | `.units` | `{stream: unit}` |
    | `.flags` | quality-flag bits, same shape as `.data` |
    | `.spec_hash` | the hash of the spec that made it |
    | `.provenance` | one JSON-able record per step |
    | `.report` | JSON-able summary: hash, version, fit report, flag counts |
    | `.to_dict()` | everything except the data, as JSON-able dict |
    """)
    return


@app.cell
def _(Flag, case, json, key, pd, result):
    result_json = json.dumps(result.to_dict())
    result_round_trip = json.loads(result_json)["spec_hash"] == key
    _spikes = case.fault_mask("spike", "flow")
    _outlier = result.signal.has_flag(Flag.OUTLIER)["flow"]
    run_summary = pd.DataFrame(
        [
            {"check": "index time zone", "value": str(result.data.index.tz)},
            {"check": "units", "value": str(result.units)},
            {"check": "flags shape", "value": str(result.flags.shape)},
            {"check": "result.spec_hash == spec_hash", "value": str(result_round_trip)},
            {"check": "steps in provenance", "value": str(len(result.provenance))},
            {"check": "to_dict() as JSON, characters", "value": str(len(result_json))},
            {"check": "spikes injected", "value": str(int(_spikes.sum()))},
            {"check": "spikes caught", "value": str(int((_outlier & _spikes).sum()))},
            {"check": "false alarms", "value": str(int((_outlier & ~_spikes).sum()))},
        ]
    )
    run_summary
    return result_round_trip, run_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The orchestrator may hold a plain DataFrame and a units dict rather than a
    `SignalFrame`. Bind them as a pair, `(df, units)`. A DataFrame on its own
    also works, but then the result has no units.
    """)
    return


@app.cell
def _(fs, mo, params, raw, result):
    from_pair = fs.run(
        params.spec, {"historian": (raw.values, raw.units)}, write_outputs=False
    )
    pair_matches = (
        from_pair.data.equals(result.data) and from_pair.units == result.units
    )
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"`(df, units)` gives the same values and units as the SignalFrame: "
        f"**{pair_matches}** (units `{from_pair.units}`)"
    )
    return (pair_matches,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Storing a fitted pipeline

    `Tukey` learns its limits from the training data. `fs.fit` returns a
    `FittedPipeline`, and `to_payload()` turns it into a plain dict: the spec,
    its hash and each step's state. Everything in it is JSON except `blobs`,
    which holds bytes for steps whose state can't be JSON (none of the steps
    here, so it is empty).

    Below, a dict stands in for the orchestrator's store, like its memory
    backend. The payload goes in as JSON text, keyed by the spec hash.
    """)
    return


@app.cell
def _(fs, json, params, raw):
    fitted = fs.fit(params.spec, {"historian": raw})
    payload = fitted.to_payload()
    store = {}
    store[f"fits/{payload['spec_hash']}"] = json.dumps(payload)
    sorted(payload)
    return fitted, payload, store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A day later, new data arrives. The orchestrator loads the payload and
    applies the stored fit to it. `from_payload` checks that the spec still
    matches its hash, so a state can't be paired with the wrong recipe. The
    result is the same as from the pipeline fitted the day before.

    Ray sends work to other processes by pickling it. Fitted pipelines and
    `SignalFrame`s pickle, so a worker can run the fit as well.
    """)
    return


@app.cell
def _(fitted, fs, json, key, pickle, store, synthetic):
    newer_case = synthetic.plant(seed=4, periods=288, start="2025-01-08").with_spikes(
        "flow", count=4, size=40
    )
    newer = newer_case.signal().select(["flow", "pressure"])

    restored = fs.FittedPipeline.from_payload(json.loads(store[f"fits/{key}"]))
    on_newer = restored.transform({"historian": newer})
    restored_matches = on_newer.equals(fitted.transform({"historian": newer}))

    on_worker = pickle.loads(pickle.dumps(fitted))
    pickled_matches = on_worker.transform({"historian": newer}).equals(on_newer)
    return newer, newer_case, on_newer, pickled_matches, restored_matches


@app.cell
def _(COLORS, Flag, newer, newer_case, on_newer, plt, style):
    _caught = on_newer.has_flag(Flag.OUTLIER)["flow"]
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.plot(newer.values["flow"], color=COLORS["fault"], lw=0.8, label="new day, raw")
    _ax.plot(
        on_newer.values["flow"],
        color=COLORS["signal"],
        lw=0.8,
        label="cleaned with the stored fit",
    )
    _ax.plot(
        newer.values["flow"][_caught],
        "o",
        ms=4,
        mfc="none",
        color=COLORS["fault"],
        label="flagged",
    )
    style(_ax, "flow [m³/h]")
    _ax.legend(
        frameon=False, ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0)
    )
    _fig.tight_layout()
    spikes_on_newer = int((_caught & newer_case.fault_mask("spike", "flow")).sum())
    _fig
    return (spikes_on_newer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Errors and progress a ledger can record

    Every flex-signal error is a `SignalError`. Its `field` is a dotted path
    into the spec, such as `steps.1.k` for the second step's `k`, so a ledger
    entry (or a form) can point at the exact option. Errors pickle too, so one
    raised on a Ray worker arrives intact.
    """)
    return


@app.cell
def _(fs, json, pd, pickle, raw, spec):
    def run_command(spec_like, data):
        """What a command wrapper might write to its ledger."""
        try:
            fs.run(spec_like, data, write_outputs=False)
        except fs.SignalError as error:
            _error = pickle.loads(pickle.dumps(error))  # as if from a worker
            return {
                "status": "failed",
                "error": type(_error).__name__,
                "field": _error.field,
                "message": str(_error),
            }
        return {"status": "done", "error": None, "field": None, "message": ""}

    _bad_k = json.loads(json.dumps(fs.dump_spec(spec)))
    _bad_k["steps"][1]["k"] = -1
    ledger = pd.DataFrame(
        {
            "good spec": run_command(spec, {"historian": raw}),
            "negative Tukey k": run_command(_bad_k, {"historian": raw}),
            "misspelled slot": run_command(spec, {"histrian": raw}),
        }
    ).T
    ledger
    return (ledger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `progress` is called as `progress(fraction, message)` after each step. An
    orchestrator's progress bar takes the same two values.
    """)
    return


@app.cell
def _(fs, params, pd, raw):
    seen = []
    fs.run(
        params.spec,
        {"historian": raw},
        write_outputs=False,
        progress=lambda frac, msg: seen.append({"fraction": frac, "message": msg}),
    )
    progress_table = pd.DataFrame(seen)
    progress_table
    return (progress_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spec versions

    A study config may be stored for years, so every spec says which version
    of the spec format it is written in: `schema_version`. This release writes
    the version in `fs.SCHEMA_VERSION`, 0.1.0, the first frozen schema.

    A spec from a newer flex-signal can't be read safely by an older one, so
    it is refused with a message that says what to do. The params model above
    gives the same refusal, because validating a spec checks its version.
    """)
    return


@app.cell
def _(CleanSignalsParams, ValidationError, fs, mo, spec):
    _newer_spec = {**fs.dump_spec(spec), "schema_version": "0.2.0"}
    try:
        fs.load_spec(_newer_spec)
    except fs.SignalConfigError as _error:
        version_error = {"field": _error.field, "message": str(_error)}
    try:
        CleanSignalsParams.model_validate(
            {"spec": _newer_spec, "output_dataset": "historian_clean"}
        )
    except ValidationError as _error:
        params_version_error = _error.errors()[0]["msg"]
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"`fs.load_spec`:\n\n> {version_error['message']}\n\n"
        f"The params model:\n\n> {params_version_error}"
    )
    return params_version_error, version_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Going the other way, later releases will upgrade older specs as they
    load them. When the schema changes, the release adds a function to
    `fs.spec.MIGRATIONS` that turns a spec of the old version into the new
    one, and `fs.load_spec` applies them in turn. So a study spec stored
    today at 0.1.0 keeps loading after an upgrade. There are no migrations
    yet, because 0.1.0 is the first release of the schema.
    """)
    return


@app.cell
def _(fs, mo):
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"`fs.SCHEMA_VERSION` → `{fs.SCHEMA_VERSION}`; "
        f"`fs.spec.MIGRATIONS` → `{fs.spec.MIGRATIONS}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next

    - The explanation page *The spec* covers inputs, outputs, versions and
      errors in more detail.
    - Tutorial 3 shows saving a fitted pipeline as files, and tutorial 4 binds
      several named inputs at once.
    """)
    return


@app.cell
def _(
    fs,
    key,
    ledger,
    outline_step_fields,
    pair_matches,
    params_round_trip,
    params_version_error,
    payload,
    pickled_matches,
    progress_table,
    restored_matches,
    result,
    result_round_trip,
    run_summary,
    same_hash_any_order,
    spec,
    spikes_on_newer,
    step_kinds,
    tukey_k,
    version_error,
):
    # The tutorial's claims, checked.
    assert spec.inputs[0].name == "historian" and spec.inputs[0].kind == "frame"
    assert spec.schema_version == fs.SCHEMA_VERSION == "0.1.0"
    assert same_hash_any_order and params_round_trip
    assert outline_step_fields == ["kind", "name"]
    assert {"range", "tukey", "hampel", "interpolate"} <= set(step_kinds)
    assert tukey_k.get("exclusiveMinimum") == 0
    assert str(result.data.index.tz) == "UTC"
    assert set(result.units) == set(result.data.columns)
    assert result.flags.shape == result.data.shape
    assert result_round_trip and result.spec_hash == key
    assert len(result.provenance) == 4
    _summary = dict(zip(run_summary["check"], run_summary["value"], strict=True))
    assert _summary["spikes caught"] == _summary["spikes injected"] == "6"
    assert _summary["false alarms"] == "0"
    assert pair_matches
    assert payload["blobs"] == {} and payload["spec_hash"] == key
    assert restored_matches and pickled_matches
    assert spikes_on_newer == 4
    assert ledger.loc["good spec", "status"] == "done"
    assert ledger.loc["negative Tukey k", "field"] == "steps.1.k"
    assert ledger.loc["misspelled slot", "field"] == "inputs"
    assert list(progress_table["fraction"]) == sorted(progress_table["fraction"])
    assert progress_table["fraction"].iloc[-1] == 1.0
    assert version_error["field"] == "schema_version"
    assert "newer than this flex-signal release" in version_error["message"]
    assert "Upgrade flex-signal" in params_version_error
    assert fs.spec.MIGRATIONS == {}
    return


if __name__ == "__main__":
    app.run()
