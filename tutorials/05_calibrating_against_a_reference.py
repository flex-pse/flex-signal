import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import flexsignal as fs
    from flexsignal.core import Flag, evaluate_condition
    from flexsignal.testing import synthetic

    return Flag, evaluate_condition, fs, json, mo, np, pd, plt, synthetic


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
    # 5. Calibrating A against an intermittent reference B

    Facilities commonnly have many sensors, some sensors might be measuring
    all the time, but have errors, while accurate sensors might not be always
    online. flex-signal is designed to help correct live sensors, by filtering out
    when good sensor data is available.

    For example:

    > Meter **A** always reports, but it is miscalibrated, and its error
    > changes over time. Meter **B** measures the same flow accurately, but
    > only some of the time, and only when valve **C** is shut and pump **D**
    > is off. Correct A using B.

    The synthetic `abcd` case has exactly this shape, and it knows the true
    flow, so every answer below is scored against the truth:

    - **A** reads 0.85 × true + 6 m³/h, plus noise.
    - **B** reports for one hour in every six. It reads 20 m³/h high while C
      is open, and 0 while D runs.
    - **C** and **D** switch on and off in blocks of 1-4 hours.
    """)
    return


@app.cell
def _(np, synthetic):
    case = synthetic.abcd(seed=0)
    raw = case.signal()
    true_a = case.underlying["A"]

    def rmse(values, truth=true_a):
        """RMS error against the true flow, over the rows that have a value."""
        return float(np.sqrt(np.nanmean((values - truth).to_numpy() ** 2)))

    return case, raw, rmse, true_a


@app.cell
def _(COLORS, case, plt, raw, style, true_a):
    _day = slice("2025-01-12", "2025-01-13")
    _obs = raw.values.loc[_day]
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    for _name, _color, _label in (
        ("C", COLORS["fault"], "C open"),
        ("D", COLORS["truth"], "D running"),
    ):
        _ax.fill_between(
            _obs.index,
            0,
            1,
            where=_obs[_name] == 1,
            transform=_ax.get_xaxis_transform(),
            color=_color,
            alpha=0.15,
            label=_label,
        )
    _ax.plot(true_a.loc[_day], color=COLORS["truth"], lw=2, label="true flow")
    _ax.plot(_obs["A"], color=COLORS["signal"], lw=0.8, label="A")
    _bad = case.any_fault("B").loc[_day]
    _ax.plot(_obs["B"][~_bad], ".", ms=4, color=COLORS["filled"], label="B, good")
    _ax.plot(_obs["B"][_bad], "x", ms=4, color=COLORS["fault"], label="B, wrong")
    style(_ax, "flow [m³/h]").set_title("Two days of A and B", loc="left", fontsize=10)
    _ax.legend(
        frameon=False, ncol=6, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08)
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A sits well below the true flow. B is right when it reports, except in
    the shaded periods: high while C is open, and 0 while D runs.

    ## The core call

    Here is the whole job in flex-signal, before we take it apart. Describe
    the step, then fit it and apply it to the data:
    """)
    return


@app.cell
def _(fs, raw):
    calibrate = fs.correct.Calibrate(
        target="A",  # the stream to correct
        reference="B",  # the trusted stream
        fit_when="C == 0 and D == 0 and B > 0",  # when B can be trusted
    )
    fitted, out = fs.Pipeline([calibrate]).fit_transform(raw)

    # The result is a new stream, A_clean, next to the originals.
    out.to_pandas().loc["2025-01-12 06:45":"2025-01-12 07:15", ["A", "B", "A_clean"]]
    return fitted, out


@app.cell
def _(fitted, fs, mo, out, raw, rmse):
    _metrics = fitted.states[0]["metrics"]
    _first = fs.core.flag_names(int(out.flags["A_clean"].loc["2025-01-12 06:45"]))
    _later = fs.core.flag_names(int(out.flags["A_clean"].loc["2025-01-12 07:15"]))
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    Until 07:00, B is reporting and trusted, so `A_clean` takes B's own
    reading (flags: {_first or "none"}). From 07:00, B is silent, and
    `A_clean` is A run through the fitted calibration (flags:
    {", ".join(_later)}). The fit used **{_metrics["rows"]}** trusted rows
    (R² = {_metrics["r2"]:.3f}), and `A_clean` is off by
    **{rmse(out.values["A_clean"]):.2f} m³/h** over the fortnight, against
    {rmse(raw.values["A"]):.1f} m³/h for raw A.

    `fitted` is the fitted pipeline: the spec plus the calibration's
    coefficients, saved as JSON with `fitted.save(...)` and applied to new
    data with `fitted.transform(...)`. The rest of this tutorial looks at
    what each option of `Calibrate` does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: say when B can be trusted

    `Calibrate` learns a mapping from A to B. It fits only on the **trusted
    rows**: the rows where B has a value **and** `fit_when` holds.
    `fit_when` is a condition over stream names, the same language as
    `Mask` and `Expression` use.

    Try each choice of `fit_when` below and watch the error of the corrected
    A:
    """)
    return


@app.cell
def _(mo):
    FIT_WHEN = {
        "every B value (no fit_when)": None,
        "B > 0": "B > 0",
        "D == 0 and B > 0": "D == 0 and B > 0",
        "C == 0 and D == 0 and B > 0": "C == 0 and D == 0 and B > 0",
    }
    # Notebook control (marimo): a dropdown to choose fit_when.
    fit_when_choice = mo.ui.dropdown(
        options=list(FIT_WHEN), value="every B value (no fit_when)", label="fit_when"
    )
    fit_when_choice
    return FIT_WHEN, fit_when_choice


@app.cell
def _(FIT_WHEN, evaluate_condition, fs, pd, raw, rmse):
    def calibrated(**options):
        """Fit Calibrate on the raw data and return (fitted, corrected frame)."""
        step = fs.correct.Calibrate(target="A", reference="B", **options)
        return fs.Pipeline([step]).fit_transform(raw)

    _rows = []
    fit_when_outputs = {}  # choice -> (corrected A, rows used for the fit)
    for _label, _condition in FIT_WHEN.items():
        _fitted, _out = calibrated(fit_when=_condition, prefer_reference=False)
        _metrics = _fitted.states[0]["metrics"]
        _used = raw.values["B"].notna()
        if _condition is not None:
            _used &= evaluate_condition(raw, _condition)
        fit_when_outputs[_label] = (_out.values["A_clean"], _used)
        _rows.append(
            {
                "fit_when": _label,
                "rows fitted": _metrics["rows"],
                "fit R²": round(_metrics["r2"], 3),
                "error of corrected A [m³/h]": round(rmse(_out.values["A_clean"]), 2),
            }
        )
    fit_when_table = pd.DataFrame(_rows).set_index("fit_when")
    raw_error = rmse(raw.values["A"])
    return calibrated, fit_when_outputs, fit_when_table, raw_error


@app.cell
def _(fit_when_choice, fit_when_table, mo, raw_error):
    _row = fit_when_table.loc[fit_when_choice.value]
    # Display only (marimo): render the result as formatted text.
    mo.vstack(
        [
            mo.md(
                f"With **{fit_when_choice.value}**, the fit uses "
                f"**{_row['rows fitted']}** rows and the corrected A is off by "
                f"**{_row['error of corrected A [m³/h]']} m³/h**. Raw A is off by "
                f"{raw_error:.1f} m³/h."
            ),
            fit_when_table,
        ]
    )
    return


@app.cell
def _(COLORS, fit_when_choice, fit_when_outputs, plt, raw, rmse, style, true_a):
    _day = slice("2025-01-12", "2025-01-13")
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    for _ax, (_label, (_clean, _used)) in zip(
        _axes.flat, fit_when_outputs.items(), strict=True
    ):
        _b = raw.values["B"].loc[_day]
        _in_fit = _used.loc[_day]
        _ax.plot(true_a.loc[_day], color=COLORS["truth"], lw=2.5, label="true flow")
        _ax.plot(
            raw.values["A"].loc[_day],
            color=COLORS["signal"],
            lw=0.6,
            alpha=0.5,
            label="raw A",
        )
        _ax.plot(_clean.loc[_day], color=COLORS["filled"], lw=1, label="A_clean")
        _ax.plot(_b[_in_fit], ".", ms=4, color=COLORS["filled"], label="B, fitted on")
        _ax.plot(_b[~_in_fit], "x", ms=4, color=COLORS["fault"], label="B, left out")
        _chosen = _label == fit_when_choice.value
        style(_ax).set_title(
            f"{_label}\nerror over the fortnight {rmse(_clean):.1f} m³/h",
            loc="left",
            fontsize=9,
            fontweight="bold" if _chosen else "normal",
        )
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("flow [m³/h]", color="#4a5568")
    _fig.autofmt_xdate()
    _fig.tight_layout(rect=(0, 0, 1, 0.94))
    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, frameon=False, ncol=5, fontsize=8, loc="upper left")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The four panels show the same two days as before, corrected with each
    choice of `fit_when` (the selected one has a bold title). Fitting on
    every B value makes A **worse than not calibrating at all**: the zeros
    and the high readings drag the line off, and `A_clean` ends up with the
    wrong gain, roughly right at peak flow but far too low at low flow.

    `B > 0` drops the zeros, and the error falls to about 3 m³/h, but the
    high readings while C is open still pull `A_clean` above the true flow
    at peak. Adding `D == 0` changes nothing here: B reads 0 whenever D
    runs, so `B > 0` had already dropped those rows. On real data, keep the
    clause anyway if D can also make B wrong without zeroing it. Only
    `C == 0` removes the high readings, and with the full condition the
    corrected A is as accurate as A's own noise allows.

    The fit R² tells the same story without knowing the truth. That is the
    number to watch on real data, where there is no truth to compare with.

    A few more helpers are useful in `fit_when`:

    - `held(C == 0, "30min")`: C has been shut for at least 30 minutes, to
      skip the transient after it closes;
    - `is_raw(B)` and `flagged(B, "OUTLIER")`: read the flags that earlier
      cleaning steps set on B.

    ## Step 2: where the output comes from

    By default `Calibrate` writes a new stream, `A_clean`, and
    **prefers the reference**: on the trusted rows it takes B's own reading,
    because a good measurement beats a prediction. Every other row is the
    calibrated A, flagged `CORRECTED`.
    """)
    return


@app.cell
def _(Flag, calibrated, pd, rmse):
    TRUSTED = "C == 0 and D == 0 and B > 0"
    fitted_static, corrected = calibrated(fit_when=TRUSTED)
    _flags = corrected.flags["A_clean"]
    _from_b = (_flags & int(Flag.CORRECTED)) == 0
    sources = pd.DataFrame(
        {
            "rows": [int(_from_b.sum()), int((~_from_b).sum())],
            "error [m³/h]": [
                round(rmse(corrected.values["A_clean"][_from_b]), 2),
                round(rmse(corrected.values["A_clean"][~_from_b]), 2),
            ],
        },
        index=["B itself (trusted rows)", "calibrated A (CORRECTED)"],
    )
    sources
    return TRUSTED, fitted_static, sources


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The rows taken from B are more accurate than the calibrated rows, since B
    is the better meter. Set `prefer_reference=False` to use the prediction
    everywhere.

    ## Step 3: when A drifts

    Real meters drift. Here is the same plant, but A's offset now grows by
    2 m³/h a day. A single calibration, fitted once, can't match both the
    start and the end of the fortnight. With
    `refit={"rolling": "1D"}`, `Calibrate` refits for each day on the
    trusted rows of the window ending with that day.

    The window width is a trade-off. A short window follows the drift
    closely but fits on fewer rows, so it is noisier. Choose a width:
    """)
    return


@app.cell
def _(mo):
    # Notebook control (marimo): a dropdown to choose the rolling window.
    window_choice = mo.ui.dropdown(
        options=["1D", "2D", "4D", "7D"], value="1D", label="rolling window"
    )
    window_choice
    return (window_choice,)


@app.cell
def _(TRUSTED, fs, np, pd, synthetic):
    drift_case = synthetic.abcd(seed=0, drift_per_day=2.0)
    drift_raw = drift_case.signal()
    drift_truth = drift_case.underlying["A"]

    def drift_run(refit):
        """Calibrate the drifting data; return the corrected A."""
        step = fs.correct.Calibrate(
            target="A",
            reference="B",
            fit_when=TRUSTED,
            refit=refit,
            prefer_reference=False,
        )
        return fs.Pipeline([step]).fit_transform(drift_raw)[1].values["A_clean"]

    drift_results = {"static": drift_run("static")}
    for _window in ["1D", "2D", "4D", "7D"]:
        drift_results[_window] = drift_run({"rolling": _window})
    drift_table = pd.DataFrame(
        {
            "error [m³/h]": {
                k: round(float(np.sqrt(np.mean((v - drift_truth) ** 2))), 2)
                for k, v in drift_results.items()
            }
        }
    )
    drift_table.index.name = "refit"
    return drift_case, drift_results, drift_table, drift_truth


@app.cell
def _(
    COLORS,
    drift_results,
    drift_table,
    drift_truth,
    mo,
    plt,
    style,
    window_choice,
):
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    for _key, _color, _label in (
        ("static", COLORS["fault"], "static"),
        (window_choice.value, COLORS["signal"], f"rolling {window_choice.value}"),
    ):
        _error = (drift_results[_key] - drift_truth).resample("6h").mean()
        _ax.plot(_error, color=_color, lw=1.4, label=_label)
    _ax.axhline(0, color=COLORS["truth"], lw=1)
    style(_ax, "error [m³/h], 6-hour mean").set_title(
        "Corrected A minus the true flow", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    # Display only (marimo): show the plot beside the table.
    mo.hstack([_fig, drift_table], widths=[3, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The static fit is right only in the middle of the fortnight; its error
    runs from too low to too high as A drifts. It also swings every day: to
    fit the drift as best it can, the static fit got the gain wrong, so its
    error depends on the flow. The rolling fits stay close to zero. Wider windows average more rows, but they lag the drift more.

    A window must also hold enough reference data to fit a **gain**, not just
    an offset: several reference periods at different flows. The first day's
    window reaches back before the data starts and holds only one hour of B,
    when the flow barely changes, so its slope is poorly pinned down. A
    window of two days or more avoids that, at the price of a little more
    lag. Days whose window has fewer than `min_rows` trusted rows (10 by
    default) use the calibration fitted on the training data.

    ## The whole recipe

    One pipeline: hygiene on B, the calibration with a 2-day rolling refit,
    and the gaps in A filled by length, short ones by interpolation and long
    ones by a Kalman smoother that knows the daily cycle. Imputation comes
    **after** calibration, so the imputer fills corrected values. We add some
    2-hour dropouts to A to give it something to do; they also remove some
    reference rows, which is where the wider window earns its keep.
    """)
    return


@app.cell
def _(TRUSTED, drift_case, fs, json, np):
    gappy = drift_case.with_gaps("A", count=6, length=24)
    pipeline = fs.Pipeline(
        [
            fs.clean.Range("B", min=0, name="b_hygiene"),
            fs.correct.Calibrate(
                target="A",
                reference="B",
                fit_when=TRUSTED,
                model="sklearn:HuberRegressor",
                refit={"rolling": "2D"},
            ),
            fs.impute.Tiered(
                "A_clean",
                tiers=[
                    {"max_gap": "15min", "method": "interpolate"},
                    {
                        "max_gap": None,
                        "method": "kalman",
                        "options": {"seasonal": [{"period": "1D", "harmonics": 2}]},
                    },
                ],
            ),
        ],
        name="a_cleanup",
    )
    spec_text = json.dumps(fs.dump_spec(pipeline.spec), indent=2)
    result = fs.run(spec_text, gappy.signal())
    final = result.signal.values["A_clean"]
    final_error = float(np.sqrt(np.mean((final - gappy.underlying["A"]) ** 2)))
    return final, final_error, gappy, result, spec_text


@app.cell
def _(final, final_error, gappy, mo, result, spec_text):
    _counts = result.signal.flag_counts()["A_clean"]["flags"]
    _title = f"The spec: {len(spec_text.splitlines())} lines of JSON"
    # Display only (marimo): text above a collapsible section.
    mo.vstack(
        [
            mo.md(
                f"`A_clean` has a value at **{int(final.notna().sum())}** of "
                f"{len(final)} rows and is off by **{final_error:.2f} m³/h**; raw A "
                f"was off by "
                f"{float(((gappy.observed['A'] - gappy.underlying['A']) ** 2).mean() ** 0.5):.1f}. "
                f"Flag counts: {_counts}."
            ),
            mo.accordion({_title: mo.md(f"```json\n{spec_text}\n```")}),
        ]
    )
    return


@app.cell
def _(
    drift_table,
    final,
    final_error,
    fit_when_table,
    fitted_static,
    fs,
    out,
    raw_error,
    sources,
):
    # These checks run when CI runs the tutorial as a script.
    _flags = out.flags["A_clean"]
    assert _flags.loc["2025-01-12 06:45"] == 0  # B's own reading
    assert _flags.loc["2025-01-12 07:15"] == fs.Flag.CORRECTED  # calibrated A
    _errors = fit_when_table["error of corrected A [m³/h]"]
    assert _errors.iloc[0] > raw_error  # no fit_when: worse than raw A
    assert _errors.is_monotonic_decreasing  # each clause helps
    assert _errors.iloc[-1] < 1.5  # the full condition: down to the noise
    assert fitted_static.states[0]["metrics"]["r2"] > 0.99
    assert sources["error [m³/h]"].iloc[0] < sources["error [m³/h]"].iloc[1]
    _drift = drift_table["error [m³/h]"]
    assert _drift["static"] > 5 and _drift["1D"] < 2
    assert _drift["1D"] < _drift["7D"] < _drift["static"]
    assert final.notna().all() and final_error < 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    Tutorial 06 measures which imputer suits which gap length, which is how
    you choose the tiers used above.
    """)
    return


if __name__ == "__main__":
    app.run()
