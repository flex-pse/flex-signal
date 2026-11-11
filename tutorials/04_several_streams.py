import marimo

__generated_with = "0.24.0"
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
    from flexsignal.core import Flag
    from flexsignal.testing import synthetic

    return Flag, fs, json, mo, np, pd, plt, synthetic


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
    # 4. Several streams: aligning, masking and conditional rules

    Real cleaning rarely involves one stream on its own. Data comes from
    several systems with different clocks, some periods shouldn't be judged
    at all, and many rules involve more than one stream: "this reading is
    wrong *when* the pump is running and the pressure disagrees with the
    flow".

    This tutorial works through those cases on two sources:

    - **SCADA**: `flow`, `pressure` and `pump_on`, every 5 minutes. The pump
      stops twice during the week.
    - **A reference meter**: a precise flow meter downstream, with its own
      logger. Water takes **10 minutes** to reach it, and its logger stamps
      readings **2 minutes past** each 5-minute mark. It only reports for two
      hours a day.
    """)
    return


@app.cell
def _(pd, synthetic):
    case = (
        synthetic.plant(seed=3, pump_off_periods=2)
        .with_spikes("flow", count=4)
        .with_out_of_range("pressure", count=3, value=3.0)  # plausible, but wrong
    )
    scada = case.observed[["flow", "pressure", "pump_on"]]
    reference = case.observed[["reference_flow"]].dropna()
    # 10 minutes of transport delay, plus the logger's 2-minute offset.
    reference = reference.set_axis(reference.index + pd.Timedelta("12min"))
    return case, reference, scada


@app.cell
def _(mo, reference, scada):
    # Display only (marimo): show these outputs side by side.
    mo.hstack(
        [
            mo.vstack([mo.md("**SCADA**"), scada.head(4)]),
            mo.vstack([mo.md("**Reference meter**"), reference.head(4)]),
        ],
        widths="equal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two named inputs

    A spec can declare **input slots**. The caller fills each slot by name,
    and flex-signal joins them on time. A joined row where a stream had no
    sample is flagged `MISSING` and `NO_SAMPLE`: the row exists only because
    the other source has a sample at that time.
    """)
    return


@app.cell
def _(fs, reference, scada):
    INPUTS = [
        {"kind": "frame", "name": "scada", "description": "5-minute SCADA export"},
        {"kind": "frame", "name": "reference", "description": "reference meter log"},
    ]
    DATA = {"scada": scada, "reference": reference}

    def run(*steps):
        """Run steps on both inputs and return the SignalFrame."""
        return fs.run(
            fs.Pipeline(list(steps), name="demo", inputs=INPUTS).spec, DATA
        ).signal

    joined = run()
    joined.to_pandas(include_flags=True).loc["2025-01-06 09:00":"2025-01-06 09:25"]
    return DATA, INPUTS, joined, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: `Align` puts every stream on one grid

    `Align` builds a regular grid and gives each stream a value at each grid
    point. It keeps a sample that is exactly on the grid, and otherwise takes
    the nearest sample within `tolerance`. Moved values are flagged
    `RESAMPLED`.

    `tolerance` sets how far a sample may be moved. The reference meter's
    readings are 2 minutes off the grid, so `tolerance="2min"` places each
    one at exactly one grid point. With a looser tolerance, a reading at the
    edge of its reporting window could also be copied to the next grid point.
    """)
    return


@app.cell
def _(fs, mo, reference, run):
    aligned = run(fs.select.Align(freq="5min", tolerance="2min"))
    _counts = aligned.flag_counts()["reference_flow"]
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"The grid has **{len(aligned)}** points. The reference meter's "
        f"**{len(reference)}** readings land on **{_counts['valid']}** of them, each "
        f"flagged `RESAMPLED`; its other points are `MISSING`."
    )
    return (aligned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: find the delay, then `Shift` it out

    The reference meter sees the same water 10 minutes later. In practice
    you'd have to find that delay, and the data can tell you. Slide the
    reference back by different amounts and compare it with the SCADA flow:
    the difference is smallest at the right delay. Clean the spikes first so
    they don't dominate the comparison.

    The minimum is shallow, because flow changes slowly: 10 minutes of delay
    shifts it by only about 1 m³/h, the same as the meter noise. A week of
    data is enough to see it, but check the answer is physically sensible,
    such as the pipe volume divided by the flow, before relying on it.
    """)
    return


@app.cell
def _(fs, np, pd, run):
    _clean = run(
        fs.select.Align(freq="5min", tolerance="2min"),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
    ).values
    _rows = []
    for _minutes in range(-20, 25, 5):
        _diff = _clean["flow"] - _clean["reference_flow"].shift(-_minutes // 5)
        _rows.append(
            {
                "delay [min]": _minutes,
                "RMS difference": float(np.sqrt(np.nanmean(_diff**2))),
            }
        )
    lag_scan = pd.DataFrame(_rows)
    best_delay = int(lag_scan.loc[lag_scan["RMS difference"].idxmin(), "delay [min]"])
    return best_delay, lag_scan


@app.cell
def _(COLORS, best_delay, lag_scan, plt, style):
    _fig, _ax = plt.subplots(figsize=(6, 2.6))
    _ax.plot(
        lag_scan["delay [min]"],
        lag_scan["RMS difference"],
        "o-",
        color=COLORS["signal"],
    )
    _ax.axvline(best_delay, color=COLORS["fault"], ls="--", lw=1)
    _ax.text(
        best_delay + 1,
        lag_scan["RMS difference"].max() * 0.9,
        f"{best_delay} min",
        color=COLORS["fault"],
        fontsize=9,
    )
    _ax.set_xlabel("assumed delay of the reference meter [min]", color="#4a5568")
    style(_ax, "RMS difference [m³/h]")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `Shift("reference_flow", by="-10min")` moves the reference readings 10
    minutes earlier, back to when that water passed the pump. Shift needs a
    regular grid, so it comes after `Align`.

    ## Step 3: `Mask` the periods you shouldn't judge

    While the pump is off, the flow meter reads noise around zero. Half of
    those readings dip below 0, so a plain `Range(min=0)` removes them as
    faults. They aren't faults: there's simply nothing to measure.

    `Mask(keep_when="pump_on == 1")` sets the pump-off values aside **before**
    cleaning. They're flagged `EXCLUDED`, which means deliberately left out,
    not bad, and `Interpolate` never fills them.
    """)
    return


@app.cell
def _(Flag, case, fs, pd, run):
    _align = fs.select.Align(freq="5min", tolerance="2min")
    _range = fs.clean.Range("flow", min=0, max=500)
    _mask = fs.select.Mask(keep_when="pump_on == 1", streams=["flow", "pressure"])
    _fill = fs.impute.Interpolate("flow", max_gap="15min")
    _pump_off = case.observed["pump_on"] == 0
    _rows = []
    for _label, _steps in (
        ("range only", (_align, _range, _fill)),
        ("mask, then range", (_align, _mask, _range, _fill)),
    ):
        _out = run(*_steps)
        _rows.append(
            {
                "steps": _label,
                "pump-off values removed as faults": int(
                    (_out.has_flag(Flag.OUT_OF_RANGE)["flow"] & _pump_off).sum()
                ),
                "pump-off values excluded": int(
                    (_out.has_flag(Flag.EXCLUDED)["flow"] & _pump_off).sum()
                ),
                "pump-off values filled": int(
                    (_out.has_flag(Flag.IMPUTED_SHORT)["flow"] & _pump_off).sum()
                ),
            }
        )
    mask_comparison = pd.DataFrame(_rows).set_index("steps")
    mask_comparison
    return (mask_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `Range` alone, pump-off readings that dip below 0 are removed as
    faults, and `Interpolate` then fills the short holes this leaves in what
    is really an idle period. With the mask first, all of them are set aside
    and none are filled.

    ## Step 4: a rule over several streams, and why order still matters

    Pressure follows flow here, at about 4 + 0.01 × flow bar. Three pressure
    readings are wrong but plausible, at 3.0 bar, so no fixed range would
    catch them. Instead we use a rule that compares the two streams. `Expression`
    takes any condition over stream names:

    ```
    abs(pressure - (4 + 0.01 * flow)) > 0.15
    ```

    The rule assumes `flow` is right. Run it **before** cleaning flow, and
    every flow spike makes a good pressure reading look inconsistent. Run it
    **after**, and the spikes are already NaN, and a comparison with NaN is
    never true:
    """)
    return


@app.cell
def _(Flag, case, fs, pd, run):
    _align = fs.select.Align(freq="5min", tolerance="2min")
    _mask = fs.select.Mask(keep_when="pump_on == 1", streams=["flow", "pressure"])
    _flow = fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0)
    _rule = fs.clean.Expression(
        "pressure", invalid_when="abs(pressure - (4 + 0.01 * flow)) > 0.15"
    )
    _truth = case.fault_mask("out_of_range", "pressure")
    _rows = []
    for _label, _steps in (
        ("rule, then clean flow", (_align, _mask, _rule, _flow)),
        ("clean flow, then rule", (_align, _mask, _flow, _rule)),
    ):
        _flagged = run(*_steps).has_flag(Flag.MANUAL)["pressure"]
        _rows.append(
            {
                "order": _label,
                "bad pressure caught": int((_flagged & _truth).sum()),
                "good pressure removed": int((_flagged & ~_truth).sum()),
            }
        )
    rule_comparison = pd.DataFrame(_rows).set_index("order")
    rule_comparison
    return (rule_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5: `Combine` the best of both meters

    Where the trusted reference meter reports, use it. Elsewhere, use the
    SCADA meter. `Combine(how="coalesce")` takes the first stream with a value
    at each time, in the order you list them. The new stream is flagged
    `DERIVED`, and each value keeps the flags of the stream it came from, so
    you can still tell the sources apart.

    ## Step 6: `Resample` and `Window` for reporting

    Last, `Resample` averages to hourly values. A bin's flags are the OR of
    its samples' flags, so an hour that contains an excluded or removed
    value says so. `Window` picks out the two days around the pump stops.
    (`Window(last="2D")` would keep the most recent two days instead.)

    ## The whole recipe

    Here it is as one pipeline, saved as JSON and run from the JSON, with the
    data bound to the two slots by name:
    """)
    return


@app.cell
def _(DATA, INPUTS, best_delay, fs, json):
    pipeline = fs.Pipeline(
        [
            fs.select.Align(freq="5min", tolerance="2min"),
            fs.select.Shift("reference_flow", by=f"-{best_delay}min"),
            fs.select.Mask(keep_when="pump_on == 1", streams=["flow", "pressure"]),
            fs.clean.Range("flow", min=0, max=500),
            fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
            fs.clean.Expression(
                "pressure",
                invalid_when="abs(pressure - (4 + 0.01 * flow)) > 0.15",
                name="pressure_follows_flow",
            ),
            fs.impute.Interpolate(["flow", "pressure"], max_gap="15min"),
            fs.select.Combine(output="best_flow", streams=["reference_flow", "flow"]),
        ],
        name="pump_station",
        inputs=INPUTS,
    )
    spec_text = json.dumps(fs.dump_spec(pipeline.spec), indent=2)
    station = fs.run(spec_text, DATA)
    station_python = fs.run(pipeline.spec, DATA)
    same_from_json = station.signal.equals(station_python.signal)
    return same_from_json, spec_text, station


@app.cell
def _(mo, same_from_json, spec_text):
    _title = (
        f"The spec: {len(spec_text.splitlines())} lines of JSON. "
        f"Same result from JSON as from Python: {same_from_json}"
    )
    # Display only (marimo): put the output in a collapsible section.
    mo.accordion({_title: mo.md(f"```json\n{spec_text}\n```")})
    return


@app.cell
def _(Flag, fs, mo, pd, station):
    _best = station.signal
    _from_reference = (
        _best.values["reference_flow"].notna() & _best.values["best_flow"].notna()
    )
    report = fs.run(
        fs.Pipeline(
            [
                fs.select.Resample(freq="1h"),
                fs.select.Window(start="2025-01-07", end="2025-01-09"),
            ],
            name="report",
        ).spec,
        _best.select(["flow", "pressure", "best_flow"]),
    ).signal
    _hours_flagged = pd.Series(
        {
            name: int(report.has_flag(flag)["best_flow"].sum())
            for name, flag in (
                ("hours with excluded values", Flag.EXCLUDED),
                ("hours with removed values", Flag.OUTLIER | Flag.OUT_OF_RANGE),
                ("hours with filled values", Flag.IMPUTED_SHORT),
            )
        },
        name="count",
    )
    # Display only (marimo): stack these outputs one above the other.
    mo.vstack(
        [
            mo.md(
                "`best_flow` comes from the reference meter at "
                f"**{int(_from_reference.sum())}** of {len(_best)} points, and from "
                "SCADA elsewhere. The hourly report "
                f"has **{len(report)}** rows:"
            ),
            _hours_flagged.to_frame(),
        ]
    )
    return (report,)


@app.cell
def _(COLORS, Flag, case, plt, report, station, style):
    _sf = station.signal
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=False)
    _ax = _axes[0]
    _off = case.observed["pump_on"] == 0
    _ax.fill_between(
        case.observed.index,
        0,
        1,
        where=_off,
        transform=_ax.get_xaxis_transform(),
        color=COLORS["excluded"],
        label="pump off (excluded)",
    )
    _ax.plot(_sf.values["best_flow"], color=COLORS["signal"], lw=0.8, label="best_flow")
    _ref = _sf.values["reference_flow"].dropna()
    _ax.plot(
        _ref.index,
        _ref,
        ".",
        ms=3,
        color=COLORS["filled"],
        label="from the reference meter",
    )
    style(_ax, "flow [m³/h]").set_title(
        "Cleaned, combined flow for the week", loc="left", fontsize=10
    )
    _ax.legend(
        frameon=False, ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08)
    )
    _ax = _axes[1]
    _ax.fill_between(
        report.index,
        0,
        1,
        where=report.has_flag(Flag.EXCLUDED)["best_flow"],
        transform=_ax.get_xaxis_transform(),
        step="post",
        color=COLORS["fault"],
        alpha=0.2,
        label="hour includes excluded values",
    )
    _ax.step(
        report.index,
        report.values["best_flow"],
        where="post",
        color=COLORS["signal"],
        lw=1,
    )
    style(_ax, "hourly mean").set_title(
        "The hourly report, 7-8 January", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    Flag,
    aligned,
    best_delay,
    joined,
    mask_comparison,
    reference,
    rule_comparison,
    same_from_json,
):
    # These checks run when CI runs the tutorial as a script.
    assert joined.has_flag(Flag.NO_SAMPLE)["reference_flow"].sum() > 0
    assert aligned.flag_counts()["reference_flow"]["valid"] == len(reference)
    assert best_delay == 10
    assert mask_comparison.loc["range only", "pump-off values removed as faults"] > 0
    assert mask_comparison.loc["mask, then range"].tolist()[0] == 0
    assert mask_comparison.loc["mask, then range", "pump-off values filled"] == 0
    assert rule_comparison.loc["rule, then clean flow", "good pressure removed"] > 0
    assert rule_comparison.loc["clean flow, then rule"].tolist() == [3, 0]
    assert same_from_json
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    These are the pieces of the problem flex-signal was built for:
    *correct stream A using a trusted reference B, but only when C and D are
    0 and B is positive*. You can now align A and B, shift out the delay,
    and pick out the periods where the condition holds.

    The next release adds the step that **fits the correction**: `calibrate`,
    plus model-based imputation for long gaps. Its tutorials pick up from
    here.
    """)
    return


if __name__ == "__main__":
    app.run()
