import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import flexsignal as fs
    from flexsignal.core import QUALITY_FLAGS, Flag
    from flexsignal.testing import synthetic

    return Flag, QUALITY_FLAGS, fs, mdates, mo, np, pd, plt, synthetic


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
    # 2. Cleaning one stream

    A flow meter on the synthetic plant has four problems: `-999` error codes,
    a sensor that froze for three hours, spikes, and short dropouts. Here we
    apply flex-signal's cleaning steps one at a time and score each one against
    the faults we injected:

    - **caught**: injected faults the step removed;
    - **missed**: injected faults it left in;
    - **false alarms**: good values it removed.

    Then we put the steps together, and see why their order matters.
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
        .with_gaps("flow", count=1, length=12)  # one long, 1-hour dropout
    )
    raw = case.signal()
    return case, raw


@app.cell
def _(fs, pd, raw):
    def clean_with(*steps, data=raw):
        """Run a few steps and return the cleaned SignalFrame."""
        return fs.run(fs.Pipeline(list(steps)).spec, data).signal

    def score(result, flag, truth, stream="flow"):
        """Compare the values a step flagged with where faults were injected."""
        flagged = result.has_flag(flag)[stream]
        return {
            "caught": int((flagged & truth).sum()),
            "missed": int((~flagged & truth).sum()),
            "false alarms": int((flagged & ~truth).sum()),
        }

    def score_table(rows):
        return pd.DataFrame(rows).set_index("step")

    return clean_with, score, score_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: `Range` removes impossible values

    A meter can't read below 0 or far above its span, so anything outside
    the physical range is an error code or a fault. `Range` takes the limits
    in the stream's own units:
    """)
    return


@app.cell
def _(Flag, case, clean_with, fs, score, score_table):
    in_range = clean_with(fs.clean.Range("flow", min=0, max=500))
    score_table(
        [
            {
                "step": "range",
                **score(
                    in_range, Flag.OUT_OF_RANGE, case.fault_mask("out_of_range", "flow")
                ),
            }
        ]
    )
    return (in_range,)


@app.cell
def _(COLORS, in_range, plt, raw, style):
    _fig, _axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    _axes[0].plot(raw.values["flow"], color=COLORS["fault"], lw=0.8)
    style(_axes[0], "raw").set_title(
        "The -999 codes dwarf the real signal", loc="left", fontsize=10
    )
    _axes[1].plot(in_range.values["flow"], color=COLORS["signal"], lw=0.8)
    style(_axes[1], "after range").set_title(
        "After Range: the signal is visible again", loc="left", fontsize=10
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: `Stuck` finds a frozen sensor

    A frozen sensor repeats its last reading. Real flow always has some noise,
    so a run of identical values lasting longer than `min_duration` is a fault.
    `tolerance` sets how close values must be to count as identical. The
    default, 0, means exactly equal.
    """)
    return


@app.cell
def _(Flag, case, clean_with, fs, score, score_table):
    stuck_removed = clean_with(
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Stuck("flow", min_duration="1h"),
    )
    score_table(
        [
            {
                "step": "stuck",
                **score(stuck_removed, Flag.STUCK, case.fault_mask("stuck", "flow")),
            }
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: `RateOfChange`, and why order matters

    `RateOfChange` removes values that jump faster than the process can
    change. Here `max_rate=100` m³/h per hour, which is about 8 m³/h per
    5-minute sample. The real flow changes by at most about 1 per sample, and
    a spike by 40.

    Each value is compared with the **last accepted** value. So if
    `RateOfChange` runs **before** `Stuck`, it compares the first good samples
    after the frozen run with the frozen value. The real flow has drifted by
    then, so the good samples look like a jump and are removed. Run `Stuck`
    first and those samples are kept:
    """)
    return


@app.cell
def _(Flag, case, clean_with, fs, pd, score):
    _range = fs.clean.Range("flow", min=0, max=500)
    _stuck = fs.clean.Stuck("flow", min_duration="1h")
    _rate = fs.clean.RateOfChange("flow", max_rate=100, per="1h")
    rate_first = clean_with(_range, _rate, _stuck)
    stuck_first = clean_with(_range, _stuck, _rate)
    _spikes = case.fault_mask("spike", "flow")
    order_scores = pd.DataFrame(
        [
            {
                "order": "range, rate_of_change, stuck",
                **score(rate_first, Flag.RATE_OF_CHANGE, _spikes),
            },
            {
                "order": "range, stuck, rate_of_change",
                **score(stuck_first, Flag.RATE_OF_CHANGE, _spikes),
            },
        ]
    ).set_index("order")
    order_scores
    return order_scores, rate_first, stuck_first


@app.cell
def _(COLORS, Flag, case, mdates, pd, plt, rate_first, stuck_first, style):
    _stuck_mask = case.fault_mask("stuck", "flow")
    _end = _stuck_mask[_stuck_mask].index[-1]
    _span = slice(_end - pd.Timedelta("1h"), _end + pd.Timedelta("1h"))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    for _ax, _result, _title in (
        (_axes[0], rate_first, "rate_of_change before stuck"),
        (_axes[1], stuck_first, "stuck before rate_of_change"),
    ):
        _ax.plot(
            case.observed["flow"][_span],
            color=COLORS["truth"],
            lw=2.5,
            label="observed",
        )
        _ax.plot(
            _result.values["flow"][_span],
            "o-",
            ms=3,
            lw=0.9,
            color=COLORS["signal"],
            label="kept",
        )
        _wrong = _result.has_flag(Flag.RATE_OF_CHANGE)["flow"][_span]
        _ax.plot(
            case.observed["flow"][_span][_wrong],
            "x",
            ms=8,
            mew=2,
            color=COLORS["fault"],
            label="removed as a jump",
        )
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        style(_ax, "flow [m³/h]" if _ax is _axes[0] else None).set_title(
            _title, loc="left", fontsize=10
        )
    _axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    _fig.suptitle("The end of the frozen run", x=0.01, ha="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: `Hampel` catches what's left

    A Hampel filter compares each value with the **median** of a window
    centred on it, and removes it if it is more than `n_sigma` robust standard
    deviations away. The robust standard deviation is 1.4826 × the median
    absolute deviation. Medians ignore the spikes themselves, so the filter
    isn't pulled off by the faults it is looking for.

    On a flat stretch, a short window can underestimate the noise, and then
    ordinary noise gets flagged. `min_scale` puts a floor under the spread.
    Set it to the meter's noise level, which is 1 m³/h here.

    Below, Hampel runs on its own after `Range` and `Stuck`, so it has to find
    the spikes by itself. Move the controls and watch the score:
    """)
    return


@app.cell
def _(mo):
    # Notebook control (marimo): moving it reruns every cell that uses it.
    n_sigma = mo.ui.slider(2, 10, step=0.5, value=3, label="n_sigma", show_value=True)
    window = mo.ui.dropdown(["30min", "1h", "2h", "4h"], value="1h", label="window")
    use_floor = mo.ui.switch(value=False, label="min_scale = 1.0 (the noise level)")
    mo.hstack([n_sigma, window, use_floor], justify="start", gap=2)
    return n_sigma, use_floor, window


@app.cell
def _(Flag, case, clean_with, fs, n_sigma, score, use_floor, window):
    hampel_step = fs.clean.Hampel(
        "flow",
        window=window.value,
        n_sigma=n_sigma.value,
        min_scale=1.0 if use_floor.value else 0.0,
    )
    hampel_result = clean_with(
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Stuck("flow", min_duration="1h"),
        hampel_step,
    )
    hampel_score = score(hampel_result, Flag.OUTLIER, case.fault_mask("spike", "flow"))
    return hampel_result, hampel_score


@app.cell
def _(COLORS, Flag, case, hampel_result, hampel_score, mo, plt, style):
    _flagged = hampel_result.has_flag(Flag.OUTLIER)["flow"]
    _spikes = case.fault_mask("spike", "flow")
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.plot(case.observed["flow"].clip(lower=0), color=COLORS["truth"], lw=0.8)
    _ax.plot(
        case.observed["flow"][_flagged & _spikes],
        "o",
        ms=6,
        color=COLORS["filled"],
        label="spike caught",
    )
    _ax.plot(
        case.observed["flow"][_flagged & ~_spikes],
        "x",
        ms=6,
        mew=2,
        color=COLORS["fault"],
        label="false alarm",
    )
    _ax.plot(
        case.observed["flow"][~_flagged & _spikes],
        "s",
        ms=6,
        mfc="none",
        color=COLORS["fault"],
        label="spike missed",
    )
    _ax.set_ylim(40, 170)
    style(_ax, "flow [m³/h]")
    _ax.legend(
        frameon=False, ncol=3, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0)
    )
    _fig.tight_layout()
    # Display only (marimo): stack these outputs one above the other.
    mo.vstack(
        [
            mo.md(
                f"**caught {hampel_score['caught']} of 6 spikes, "
                f"{hampel_score['false alarms']} false alarms**"
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(Flag, case, clean_with, fs, pd, score):
    # The same comparison as a table, for a few settings.
    _rows = []
    for _sigma in (3, 4, 6):
        for _floor in (0.0, 1.0):
            _result = clean_with(
                fs.clean.Range("flow", min=0, max=500),
                fs.clean.Stuck("flow", min_duration="1h"),
                fs.clean.Hampel("flow", window="1h", n_sigma=_sigma, min_scale=_floor),
            )
            _rows.append(
                {
                    "n_sigma": _sigma,
                    "min_scale": _floor,
                    **score(_result, Flag.OUTLIER, case.fault_mask("spike", "flow")),
                }
            )
    hampel_sweep = pd.DataFrame(_rows)
    hampel_sweep
    return (hampel_sweep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every setting catches all six spikes, which are 40 noise levels high.
    What changes is the false alarms. A smaller `n_sigma` or no floor removes
    good data. `n_sigma=6` with `min_scale=1.0` removes nothing it shouldn't.
    With real data you don't have the answer key, so start strict, like this,
    and loosen it only if you can see faults getting through.

    ## Step 5: `Interpolate` fills short gaps, and only short ones

    After cleaning, the removed values and the original dropouts are all
    NaN. `Interpolate` fills a gap only if it is no longer than `max_gap`,
    measured between the good samples either side. Filling a long gap with a
    straight line would invent data. Filled values are flagged
    `IMPUTED_SHORT`, so they're never mistaken for measurements.
    """)
    return


@app.cell
def _(mo):
    # Notebook control (marimo): moving it reruns every cell that uses it.
    max_gap = mo.ui.dropdown(
        ["10min", "15min", "30min", "1h", "2h"], value="15min", label="max_gap"
    )
    max_gap
    return (max_gap,)


@app.cell
def _(Flag, case, clean_with, fs, max_gap, mo, np):
    filled = clean_with(
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Stuck("flow", min_duration="1h"),
        fs.clean.RateOfChange("flow", max_rate=100, per="1h"),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
        fs.impute.Interpolate("flow", max_gap=max_gap.value),
    )
    _was_filled = filled.has_flag(Flag.IMPUTED_SHORT)["flow"]
    _still_missing = filled.values["flow"].isna()
    _error = filled.values["flow"][_was_filled] - case.clean["flow"][_was_filled]
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    With `max_gap="{max_gap.value}"`: **{int(_was_filled.sum())}** values filled,
    **{int(_still_missing.sum())}** left as NaN. The filled values differ from the
    fault-free measurement by {np.sqrt(np.mean(_error**2)):.2f} m³/h (RMS); the
    meter's own noise is 1.0.
    """)
    return (filled,)


@app.cell
def _(COLORS, Flag, case, filled, mdates, pd, plt, style):
    # Label each run of injected gap samples, and find a short one and the long one.
    _gaps = case.fault_mask("gap", "flow")
    _run_id = (~_gaps).cumsum()[_gaps]
    _run_starts = _run_id.groupby(_run_id).apply(lambda run: run.index[0])
    _run_lengths = _run_id.value_counts().sort_index()
    _short = _run_starts[_run_lengths.index[_run_lengths == 2][0]]
    _long = _run_starts[_run_lengths.idxmax()]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for _ax, _start, _title in (
        (_axes[0], _short, "a 10-minute dropout: filled"),
        (_axes[1], _long, "the 1-hour dropout: left as NaN"),
    ):
        _span = slice(_start - pd.Timedelta("1h"), _start + pd.Timedelta("2h"))
        _ax.plot(
            case.clean["flow"][_span], color=COLORS["truth"], lw=2.5, label="fault-free"
        )
        _ax.plot(
            filled.values["flow"][_span],
            "o-",
            ms=2.5,
            lw=0.8,
            color=COLORS["signal"],
            label="cleaned",
        )
        _imp = filled.has_flag(Flag.IMPUTED_SHORT)["flow"][_span]
        _ax.plot(
            filled.values["flow"][_span][_imp],
            "o",
            ms=5,
            color=COLORS["filled"],
            label="filled",
        )
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        style(_ax, "flow [m³/h]" if _ax is _axes[0] else None).set_title(
            _title, loc="left", fontsize=10
        )
    _axes[0].legend(frameon=False, fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All together

    The full chain, in the order that works:

    1. `Range` first, so error codes can't distort the statistics later steps compute;
    2. `Stuck` before `RateOfChange`, so the end of a frozen run isn't taken for a jump;
    3. `RateOfChange` and `Hampel` for spikes. They overlap, which is fine: a
       value removed by one isn't seen by the next;
    4. `Interpolate` last, once every bad value is gone.
    """)
    return


@app.cell
def _(QUALITY_FLAGS, case, clean_with, fs, pd):
    final = clean_with(
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Stuck("flow", min_duration="1h"),
        fs.clean.RateOfChange("flow", max_rate=100, per="1h"),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
        fs.impute.Interpolate("flow", max_gap="15min"),
    )
    removed = final.has_flag(QUALITY_FLAGS)["flow"]
    injected = (
        case.fault_mask("out_of_range", "flow")
        | case.fault_mask("stuck", "flow")
        | case.fault_mask("spike", "flow")
    )
    pd.Series(final.flag_counts()["flow"]["flags"], name="values").to_frame()
    return final, injected, removed


@app.cell
def _(COLORS, Flag, final, plt, raw, style):
    _flow = final.values["flow"]
    _filled = final.has_flag(Flag.IMPUTED_SHORT)["flow"]
    _fig, _ax = plt.subplots(figsize=(9, 3.0))
    _ax.fill_between(
        _flow.index,
        0,
        1,
        where=_flow.isna(),
        transform=_ax.get_xaxis_transform(),
        color=COLORS["fault"],
        alpha=0.2,
        step="mid",
        label="still NaN (longer than max_gap)",
    )
    _ax.plot(
        raw.values["flow"].clip(lower=40),
        color=COLORS["fault"],
        lw=0.8,
        label="raw (-999 drawn at 40)",
    )
    _ax.plot(_flow, color=COLORS["signal"], lw=0.8, label="cleaned")
    _ax.plot(
        _flow[_filled].index,
        _flow[_filled],
        "o",
        ms=4,
        color=COLORS["filled"],
        label=f"filled ({int(_filled.sum())} values)",
    )
    _ax.set_ylim(35, 170)
    style(_ax, "flow [m³/h]")
    _ax.legend(
        frameon=False, ncol=4, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0)
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Flag, final, mo):
    _missing = final.values["flow"].isna()
    _runs = (_missing != _missing.shift()).cumsum()[_missing]
    _gaps = [
        f"{len(_run) * 5} minutes from {_run.index[0]:%a %H:%M}"
        for _, _run in _runs.groupby(_runs)
    ]
    _stuck = int(final.has_flag(Flag.STUCK)["flow"].sum())
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The green dots are the values `Interpolate` filled: the error codes, the
    spikes and the short dropouts, each one or two samples long. Two gaps stay
    empty on purpose, because they are longer than `max_gap="15min"`:
    {" and ".join(_gaps)}. One is the 1-hour dropout; the other is the
    {_stuck}-sample frozen run that `Stuck` removed. Guessing across hours of
    flow is not a job for interpolation. The imputers in a later tutorial fill
    long gaps from the other streams and the daily pattern.
    """)
    return


@app.cell
def _(Flag, case, final, hampel_sweep, injected, order_scores, removed):
    # These checks run when CI runs the tutorial as a script.
    assert removed.equals(injected), "removed exactly the injected faults"
    # Only the stuck run and the 1-hour dropout stay empty; everything else is filled.
    _left = final.values["flow"].isna()
    assert _left[case.fault_mask("stuck", "flow")].all()  # the 3-hour frozen run
    assert not (_left & ~case.any_fault("flow")).any()  # no good value left empty
    assert int(_left.sum()) == 36 + 12  # plus the 1-hour dropout, nothing else
    assert int(final.has_flag(Flag.IMPUTED_SHORT)["flow"].sum()) == 4 + 6 + 8
    assert order_scores["false alarms"].tolist()[0] > 0
    assert order_scores["false alarms"].tolist()[1] == 0
    _strict = hampel_sweep.query("n_sigma == 6 and min_scale == 1.0")
    assert _strict["false alarms"].item() == 0 and _strict["caught"].item() == 6
    assert (hampel_sweep["caught"] == 6).all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next

    The steps above are already a recipe. **Tutorial 3** turns it into a
    pipeline you can save as JSON, rerun anywhere, and fit on training data.
    """)
    return


if __name__ == "__main__":
    app.run()
