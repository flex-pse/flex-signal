import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import tempfile
    from pathlib import Path

    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    import flexsignal as fs
    from flexsignal.core import Flag
    from flexsignal.io import read_parquet
    from flexsignal.testing import synthetic

    return Flag, Path, fs, json, mo, pd, plt, read_parquet, synthetic, tempfile


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
    # 3. Pipelines, specs and fitted state

    In tutorial 2 we ran cleaning steps by hand. A **pipeline** is the same
    steps written down once, as data. That gives you:

    - a **spec**: plain JSON you can save, review in a pull request, or hand
      to flex-orchestrator to run;
    - a **hash** of the spec, so every result says exactly which recipe made it;
    - **provenance**: a record of what each step changed;
    - **fitted state** for steps that learn from training data, saved and
      reloaded without pickle.
    """)
    return


@app.cell
def _(synthetic):
    case = (
        synthetic.plant(seed=7)
        .with_out_of_range("flow", count=4)
        .with_stuck("flow", length=36)
        .with_spikes("flow", count=6, size=40)
        .with_gaps("flow", count=4, length=2)
    )
    raw = case.signal()
    return case, raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A pipeline is a list of steps

    These are the steps tutorial 2 ended with. Each step has a `kind`, and
    you can also give it a `name`, which shows up in the provenance.
    """)
    return


@app.cell
def _(fs):
    pipe = fs.Pipeline(
        [
            fs.clean.Range("flow", min=0, max=500, name="sentinels"),
            fs.clean.Stuck("flow", min_duration="1h"),
            fs.clean.RateOfChange("flow", max_rate=100, per="1h"),
            fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
            fs.impute.Interpolate("flow", max_gap="15min"),
        ],
        name="flow_cleanup",
        description="Remove error codes, frozen runs and spikes; fill short gaps.",
    )
    pipe
    return (pipe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The spec is JSON

    `fs.dump_spec` turns the pipeline into a plain dict. Options left at
    their defaults are omitted, so the JSON shows only the choices you made.
    `schema_version` says which spec format it is written in, so a future
    flex-signal can upgrade an old file.
    """)
    return


@app.cell
def _(fs, json, mo, pipe):
    spec_json = json.dumps(fs.dump_spec(pipe.spec), indent=2)
    # Display only (marimo): render the result as formatted text.
    mo.md(f"```json\n{spec_json}\n```")
    return (spec_json,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The JSON is the pipeline: loading it back and running it gives
    **bit-identical** results, flags and all.
    """)
    return


@app.cell
def _(fs, mo, pipe, raw, spec_json):
    result = fs.run(pipe.spec, raw)
    from_json = fs.run(spec_json, raw)
    identical = result.signal.equals(from_json.signal)
    # Display only (marimo): render the result as formatted text.
    mo.md(f"Python pipeline and JSON spec give identical output: **{identical}**")
    return identical, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The spec hash names the recipe

    `spec_hash` is a SHA-256 of the spec's canonical JSON. The same recipe
    always has the same hash, and any change to it, even one parameter,
    gives a new hash. Every result carries it.
    """)
    return


@app.cell
def _(fs, pd, pipe, result):
    _looser = fs.Pipeline(
        [
            *pipe.spec.steps[:3],
            fs.clean.Hampel("flow", window="1h", n_sigma=4),
            *pipe.spec.steps[4:],
        ],
        name="flow_cleanup",
    )
    hash_table = pd.DataFrame(
        [
            {"spec": "flow_cleanup", "spec_hash": fs.spec_hash(pipe.spec)[:16] + "…"},
            {
                "spec": "the result of running it",
                "spec_hash": result.spec_hash[:16] + "…",
            },
            {
                "spec": "same, but Hampel n_sigma=4",
                "spec_hash": fs.spec_hash(_looser.spec)[:16] + "…",
            },
        ]
    )
    hash_table
    return (hash_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Provenance: what each step did

    Every result keeps one record per step. The pipeline writes these
    records by comparing each step's input and output, so a step can't
    forget to report what it changed.
    """)
    return


@app.cell
def _(pd, result):
    pd.DataFrame(result.provenance)[
        ["index", "kind", "name", "samples_changed", "rows_before", "rows_after"]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mistakes are caught when the spec is loaded

    A spec is checked before any data is touched. An error names the exact
    field, like `steps.3.n_sigma`, and says what's wrong, so an orchestrator
    or a form can point at it:
    """)
    return


@app.cell
def _(fs, mo):
    _bad_specs = {
        "a negative threshold": {
            "schema_version": fs.SCHEMA_VERSION,
            "name": "typo",
            "steps": [
                {"kind": "hampel", "streams": ["flow"], "window": "1h", "n_sigma": -1}
            ],
        },
        "a misspelled step kind": {
            "schema_version": fs.SCHEMA_VERSION,
            "name": "typo",
            "steps": [{"kind": "hampl", "streams": ["flow"], "window": "1h"}],
        },
        "a duration that isn't one": {
            "schema_version": fs.SCHEMA_VERSION,
            "name": "typo",
            "steps": [
                {"kind": "stuck", "streams": ["flow"], "min_duration": "an hour"}
            ],
        },
    }
    _messages = []
    for _what, _spec in _bad_specs.items():
        try:
            fs.load_spec(_spec)
        except fs.SignalConfigError as _error:
            _messages.append(f"**{_what}**\n\n> {_error}")
    spec_errors = _messages
    # Display only (marimo): render the result as formatted text.
    mo.md("\n\n".join(spec_errors))
    return (spec_errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steps that learn: fit on training data, apply to new data

    Most cleaning steps are rules. A few **learn** from data. `Tukey` sets
    its limits from the spread of the training data: anything more than
    `k` interquartile ranges outside the middle half is an outlier. This is
    useful when you don't know a meter's physical limits.

    What matters is **which** data it learns from. Below, a meter is
    recalibrated badly halfway through the test week, and reads 70 m³/h too
    high from then on:

    - **fitted on a clean training week**, the limits come from normal
      operation, and the offset readings fall outside them;
    - **fitted on the faulty test week itself**, the bad data widens the
      limits and hides itself.
    """)
    return


@app.cell
def _(synthetic):
    train = synthetic.plant(seed=1).signal().select(["flow"])
    test_case = synthetic.plant(seed=2, start="2025-01-13").with_step(
        "flow", start=1008, size=70.0
    )
    test = test_case.signal().select(["flow"])
    return test, test_case, train


@app.cell
def _(Flag, fs, pd, test, test_case, train):
    tukey = fs.Pipeline([fs.clean.Tukey("flow", k=1.5)], name="tukey_flow")
    fitted = tukey.fit(train)
    on_new_data = fitted.transform(test)
    refitted = tukey.fit(test)
    on_itself = refitted.transform(test)

    _offset = test_case.fault_mask("step", "flow")
    tukey_comparison = pd.DataFrame(
        [
            {
                "fitted on": _label,
                "lower limit": _fit.states[0]["fences"]["flow"][0],
                "upper limit": _fit.states[0]["fences"]["flow"][1],
                "offset readings removed": int(
                    (_out.has_flag(Flag.OUTLIER)["flow"] & _offset).sum()
                ),
                "good readings removed": int(
                    (_out.has_flag(Flag.OUTLIER)["flow"] & ~_offset).sum()
                ),
            }
            for _label, _fit, _out in (
                ("clean training week", fitted, on_new_data),
                ("the faulty week itself", refitted, on_itself),
            )
        ]
    ).round(1)
    tukey_comparison
    return fitted, on_new_data, tukey_comparison


@app.cell
def _(COLORS, fitted, on_new_data, plt, style, test):
    _low, _high = fitted.states[0]["fences"]["flow"]
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.plot(
        test.values["flow"],
        color=COLORS["fault"],
        lw=0.8,
        label="test week, as measured",
    )
    _ax.plot(on_new_data.values["flow"], color=COLORS["signal"], lw=0.8, label="kept")
    for _limit in (_low, _high):
        _ax.axhline(_limit, color=COLORS["truth"], ls="--", lw=1)
    _ax.text(
        test.index[10],
        _high + 3,
        "limits fitted on the training week",
        fontsize=8,
        color="#4a5568",
    )
    style(_ax, "flow [m³/h]")
    _ax.legend(
        frameon=False, ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0)
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Tukey removes only the offset readings above its upper limit. The
    offset readings near the bottom of the daily cycle still look plausible.
    A rule that only looks at single values can't tell those are wrong. The
    calibration tutorial (coming with the next release) deals with them by
    comparing against a trusted reference meter.

    The learned limits are the step's **state**: plain JSON, kept apart from
    the spec.
    """)
    return


@app.cell
def _(fitted, mo):
    # Display only (marimo): render the result as formatted text.
    mo.md(f"`fitted.states` → `{fitted.states}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saving a fitted pipeline

    `save` writes the spec and each step's state as JSON files, with no
    pickle, so a saved pipeline is safe to load and easy to read. `load`
    checks the spec hash, so a state can't be paired with the wrong recipe.
    For passing a fitted pipeline between processes, `to_payload` gives the
    same content as one JSON-able dict.
    """)
    return


@app.cell
def _(Path, fitted, fs, json, mo, tempfile, test):
    workdir = Path(tempfile.mkdtemp(prefix="flexsignal-tutorial-"))
    saved_to = fitted.save(workdir / "tukey_flow")
    reloaded = fs.FittedPipeline.load(saved_to)
    same_after_reload = reloaded.transform(test).equals(fitted.transform(test))
    _payload = fitted.to_payload()
    same_after_payload = (
        fs.FittedPipeline.from_payload(json.loads(json.dumps(_payload)))
        .transform(test)
        .equals(fitted.transform(test))
    )
    _files = sorted(
        p.relative_to(saved_to).as_posix() for p in saved_to.rglob("*.json")
    )
    # Display only (marimo): render the result as formatted text.
    mo.md(
        "Files written:\n\n"
        + "\n".join(f"- `{name}`" for name in _files)
        + f"\n\nSame result after `load`: **{same_after_reload}**; "
        f"after a JSON round-trip of `to_payload()`: **{same_after_payload}**."
    )
    return same_after_payload, same_after_reload, workdir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs and outputs in the spec

    A spec can also say where data comes from and where results go. A
    `frame` input is a named slot that the caller fills at run time. A
    `parquet` output keeps values, flags, units and provenance together in
    one file. Relative paths resolve against the spec file's folder, so a
    folder holding a spec and its data can be moved as a whole.
    """)
    return


@app.cell
def _(fs, json, mo, pipe, raw, read_parquet, workdir):
    _spec = fs.dump_spec(pipe.spec)
    _spec["inputs"] = [{"kind": "frame", "name": "historian"}]
    _spec["outputs"] = [{"kind": "parquet", "path": "flow_clean.parquet"}]
    _spec_file = workdir / "flow_cleanup.json"
    _spec_file.write_text(json.dumps(_spec, indent=2))

    file_run = fs.run(_spec_file, {"historian": raw})
    from_disk = read_parquet(workdir / "flow_clean.parquet")
    disk_round_trip = from_disk.equals(file_run.signal)
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"Wrote `{file_run.outputs[0]}`. Read back with `read_parquet`, it is "
        f"identical to the result, flags and history included: **{disk_round_trip}**"
    )
    return (disk_round_trip,)


@app.cell
def _(
    disk_round_trip,
    hash_table,
    identical,
    result,
    same_after_payload,
    same_after_reload,
    spec_errors,
    tukey_comparison,
):
    # These checks run when CI runs the tutorial as a script.
    assert identical and same_after_reload and same_after_payload and disk_round_trip
    assert hash_table["spec_hash"][0] == hash_table["spec_hash"][1]
    assert hash_table["spec_hash"][0] != hash_table["spec_hash"][2]
    assert len(spec_errors) == 3
    assert "steps.0.n_sigma" in spec_errors[0]
    _trained, _self = tukey_comparison.to_dict("records")
    assert _trained["offset readings removed"] > _self["offset readings removed"]
    assert _trained["good readings removed"] == 0
    assert [r["kind"] for r in result.provenance][0] == "range"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next

    So far there has been one stream. **Tutorial 4** works with several:
    lining up sources with different clocks, cleaning only while the pump
    runs, and rules that involve more than one stream.
    """)
    return


if __name__ == "__main__":
    app.run()
