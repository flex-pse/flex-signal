import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import flexsignal as fs
    from flexsignal.core import Flag
    from flexsignal.eval import benchmark_imputers, hide_values, summarize
    from flexsignal.testing import synthetic

    return (
        Flag,
        benchmark_imputers,
        fs,
        hide_values,
        mo,
        np,
        pd,
        plt,
        summarize,
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
    # 6. Comparing imputers by gap length

    Tutorial 02 filled 10-minute gaps with a straight line, and tutorial 05
    filled 2-hour gaps with a Kalman smoother. Which method should fill
    which gap? It depends on the data, so this tutorial **measures** it.

    Real gaps have no ground truth. The benchmark makes gaps where the
    values *are* known: it hides runs of good samples, lets each imputer
    fill them, and compares the fill with what was there.

    The data is two weeks of the synthetic plant, with ten pump stops of
    2-6 hours. While the pump is off, the flow is 0. That is the hard part:
    nothing in the flow's own history says when the next stop comes, but the
    pressure does, because it follows the flow.
    """)
    return


@app.cell
def _(synthetic):
    case = synthetic.plant(seed=5, periods=14 * 288, pump_off_periods=10)
    clean = case.signal()
    return case, clean


@app.cell
def _(COLORS, clean, plt, style):
    _week = clean.values.loc["2025-01-06":"2025-01-12"]
    _fig, _ax = plt.subplots(figsize=(9, 2.6))
    _ax.fill_between(
        _week.index,
        0,
        1,
        where=_week["pump_on"] == 0,
        transform=_ax.get_xaxis_transform(),
        color=COLORS["excluded"],
        label="pump off",
    )
    _ax.plot(_week["flow"], color=COLORS["signal"], lw=0.8, label="flow")
    style(_ax, "flow [m³/h]").set_title(
        "The first week: a daily cycle, broken by pump stops", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The core calls

    Before the full comparison, here is the idea in a few lines. Hide some
    values whose truth we know, fill them with an imputer, and score the
    fill:
    """)
    return


@app.cell
def _(clean, fs, hide_values, np):
    # Hide three 6-hour gaps in the flow; `hidden` marks where.
    masked, hidden = hide_values(clean, ["flow"], gap="6h", count=3, seed=0)

    # An imputer is a step like any other: describe it, then fit and apply it.
    imputer = fs.impute.Regression("flow", features=["pressure"])
    _, filled = fs.Pipeline([imputer]).fit_transform(masked)

    # Score the fill against the values that were hidden.
    _where = hidden["flow"]
    error = filled.values["flow"][_where] - clean.values["flow"][_where]
    rms_error = float(np.sqrt(np.mean(error**2)))
    rms_error
    return imputer, rms_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `benchmark_imputers` does exactly that for several imputers and gap
    lengths at once, and `summarize` turns its results into a table of the
    RMS error by imputer and gap length:
    """)
    return


@app.cell
def _(benchmark_imputers, clean, fs, imputer, summarize):
    quick = benchmark_imputers(
        clean,
        {
            "linear": fs.impute.Interpolate("flow", max_gap="1D"),
            "regression": imputer,
        },
        streams=["flow"],
        gaps=["30min", "6h"],
        count=5,
    )
    quick_rmse = summarize(quick)
    quick_rmse.round(2)
    return (quick_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interpolation wins the short gaps and loses the long ones. The rest of
    this tutorial does the same with more methods and gap lengths, then
    turns the result into a tiered imputer.

    ## The candidates

    Every imputer is set up to attempt every gap (`max_gap` wide or unset),
    so the benchmark compares them at full coverage:

    - **linear** and **pchip** interpolation (`interpolate`);
    - a **Kalman** smoother of a level, with and without a daily cycle
      (`kalman`);
    - **regression on pressure**: a linear model of flow from pressure
      (`regression`);
    - **k-nearest neighbours** and **iterative** imputation on pressure and
      temperature (`knn`, `iterative`).
    """)
    return


@app.cell
def _(fs):
    DAILY = [{"period": "1D", "harmonics": 2}]
    candidates = {
        "linear": fs.impute.Interpolate("flow", max_gap="2D"),
        "pchip": fs.impute.Interpolate("flow", max_gap="2D", method="pchip"),
        "kalman": fs.impute.Kalman("flow"),
        "kalman + daily cycle": fs.impute.Kalman("flow", seasonal=DAILY),
        "regression on pressure": fs.impute.Regression("flow", features=["pressure"]),
        "knn": fs.impute.KNN("flow", features=["pressure", "temperature"]),
        "iterative": fs.impute.Iterative("flow", features=["pressure", "temperature"]),
    }
    GAPS = ["10min", "30min", "2h", "6h", "12h"]
    # One colour per candidate, used in every figure below.
    FILL_COLORS = {
        "linear": "#dd6b20",
        "pchip": "#f6ad55",
        "kalman": "#b794f4",
        "kalman + daily cycle": "#805ad5",
        "regression on pressure": "#2f855a",
        "knn": "#d53f8c",
        "iterative": "#319795",
    }
    return FILL_COLORS, GAPS, candidates


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One gap, five fills

    First, one gap by hand. We hide the flow around the start of a pump
    stop, then let five methods fill it: two that use only the flow itself,
    and three that also use the pressure (and, for knn and iterative, the
    temperature). A gap's length is measured the way
    imputers measure it, between the valid samples either side, so a
    30-minute gap on 5-minute data hides 5 samples. Pick a length:
    """)
    return


@app.cell
def _(GAPS, mo):
    # Notebook control (marimo): a dropdown to choose the gap length.
    gap_choice = mo.ui.dropdown(options=GAPS, value="6h", label="gap length")
    gap_choice
    return (gap_choice,)


@app.cell
def _(COLORS, FILL_COLORS, candidates, clean, fs, gap_choice, np, pd, plt, style):
    # Hide the samples of one gap, centred on the start of the third pump stop.
    _pump = clean.values["pump_on"].to_numpy()
    _stop = np.flatnonzero(np.diff(_pump) < 0)[2] + 1
    _n = int(pd.Timedelta(gap_choice.value) / pd.Timedelta("5min")) - 1
    _where = np.arange(_stop - _n // 2, _stop - _n // 2 + _n)
    _values = clean.values
    _values.iloc[_where, _values.columns.get_loc("flow")] = np.nan
    _masked = clean.evolve(values=_values)
    _span = slice(
        max(_where[0] - 3 * len(_where) - 12, 0), _where[-1] + 3 * len(_where) + 12
    )
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.0), sharey=True)
    _times = clean.index[_span]
    _groups = (
        ("Uses only the flow", ["linear", "kalman + daily cycle"]),
        (
            "Also uses pressure (and temperature)",
            ["regression on pressure", "knn", "iterative"],
        ),
    )
    for _ax, (_title, _names) in zip(_axes, _groups, strict=True):
        _ax.plot(
            _times,
            clean.values["flow"].iloc[_span],
            color=COLORS["truth"],
            lw=2,
            label="hidden truth",
        )
        _ax.plot(
            _times,
            _masked.values["flow"].iloc[_span],
            color=COLORS["signal"],
            lw=1,
            label="what the imputer sees",
        )
        for _name in _names:
            _, _out = fs.Pipeline([candidates[_name]]).fit_transform(_masked)
            _ax.plot(
                clean.index[_where],
                _out.values["flow"].iloc[_where],
                {"knn": "x", "iterative": "+"}.get(_name, "."),
                ms=4 if _name not in ("knn", "iterative") else 5,
                color=FILL_COLORS[_name],
                label=_name,
            )
        style(_ax).set_title(_title, loc="left", fontsize=10)
        _ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _axes[0].set_ylabel("flow [m³/h]", color="#4a5568")
    _fig.suptitle(f"One hidden {gap_choice.value} gap", x=0.01, ha="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a short gap every method does about as well as the noise allows. For
    a long one the line just joins the two ends, and the daily cycle is no
    help, because the pump stops inside the gap. The three methods on the
    right all see the pressure, so they all see the stop.

    ## The benchmark

    One gap proves little. `benchmark_imputers` uses `hide_values` to hide 6
    gaps of each length at random good places, lets every candidate fill
    them, and repeats that twice with the gaps in new places. It returns one
    row per imputer, stream, gap length and repeat; `summarize` turns that
    into a table:
    """)
    return


@app.cell
def _(GAPS, benchmark_imputers, candidates, clean, summarize):
    results = benchmark_imputers(
        clean, candidates, streams=["flow"], gaps=GAPS, count=6, repeats=2, seed=1
    )
    rmse = summarize(results)
    coverage = summarize(results, "coverage")
    rmse.round(2)
    return coverage, results, rmse


@app.cell
def _(FILL_COLORS, GAPS, plt, rmse, style):
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _dotted = {"pchip", "kalman", "iterative"}  # so overlapping lines stay visible
    for _name, _color in FILL_COLORS.items():
        _ls = ":" if _name in _dotted else "-"
        _ax.plot(
            GAPS, rmse.loc[_name], ls=_ls, marker="o", ms=3, color=_color, label=_name
        )
    _ax.set_yscale("log")
    _ax.set_xlabel("gap length", color="#4a5568")
    style(_ax, "RMS error [m³/h], log scale").set_title(
        "Error of the fill, by gap length", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the fills look like

    The table averages over many gaps. To see what each candidate actually
    put back, here are the gaps from the benchmark's first repeat: the same
    `hide_values` call with the same seed, so they are exactly the gaps that
    were scored. Pick a gap length, then one of its six gaps. The gap shown
    first is the one where linear interpolation does worst.
    """)
    return


@app.cell
def _(GAPS, mo):
    # Notebook control (marimo): a dropdown to choose the gap length.
    fill_gap_choice = mo.ui.dropdown(options=GAPS, value="6h", label="gap length")
    fill_gap_choice
    return (fill_gap_choice,)


@app.cell
def _(GAPS, candidates, clean, fill_gap_choice, fs, hide_values, np):
    # The benchmark seeds each gap length's first repeat as seed + 1000 * index.
    _masked, _hidden = hide_values(
        clean,
        ["flow"],
        gap=fill_gap_choice.value,
        count=6,
        seed=1 + 1000 * GAPS.index(fill_gap_choice.value),
    )
    fill_hidden = _hidden["flow"].to_numpy()
    fill_masked = _masked
    fills = {
        _name: fs.Pipeline([_step]).fit_transform(_masked)[1].values["flow"]
        for _name, _step in candidates.items()
    }
    # Split the hidden samples into the six separate gaps.
    _pos = np.flatnonzero(fill_hidden)
    fill_runs = np.split(_pos, np.flatnonzero(np.diff(_pos) > 1) + 1)
    _truth = clean.values["flow"].to_numpy()
    _linear = fills["linear"].to_numpy()
    worst_run = int(
        np.argmax([np.mean((_linear[r] - _truth[r]) ** 2) for r in fill_runs])
    )
    return fill_hidden, fill_masked, fill_runs, fills, worst_run


@app.cell
def _(fill_runs, mo, worst_run):
    # Notebook control (marimo): a slider to choose which gap to show.
    run_choice = mo.ui.slider(
        start=1,
        stop=len(fill_runs),
        value=worst_run + 1,
        label="which gap",
        show_value=True,
    )
    run_choice
    return (run_choice,)


@app.cell
def _(
    COLORS,
    FILL_COLORS,
    clean,
    fill_gap_choice,
    fill_masked,
    fill_runs,
    fills,
    np,
    plt,
    run_choice,
    style,
):
    _run = fill_runs[run_choice.value - 1]
    _pad = max(2 * len(_run), 24)
    _span = slice(max(_run[0] - _pad, 0), min(_run[-1] + _pad + 1, len(clean)))
    _times = clean.index[_span]
    _truth = clean.values["flow"]
    _fig, _axes = plt.subplots(2, 4, figsize=(11, 4.6), sharex=True, sharey=True)
    for _ax, (_name, _filled) in zip(_axes.flat, fills.items(), strict=False):
        _error = np.sqrt(np.mean((_filled.iloc[_run] - _truth.iloc[_run]) ** 2))
        _ax.plot(_times, _truth.iloc[_span], color=COLORS["truth"], lw=2.5)
        _ax.plot(
            _times,
            fill_masked.values["flow"].iloc[_span],
            color=COLORS["signal"],
            lw=0.8,
        )
        _ax.plot(
            clean.index[_run], _filled.iloc[_run], ".", ms=3, color=FILL_COLORS[_name]
        )
        style(_ax).set_title(
            f"{_name}\nerror {_error:.1f} m³/h", loc="left", fontsize=9
        )
        _ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    # The eighth panel holds the legend.
    _key = _axes.flat[-1]
    _key.axis("off")
    _key.plot([], [], color=COLORS["truth"], lw=2.5, label="hidden truth")
    _key.plot([], [], color=COLORS["signal"], lw=0.8, label="what the imputer sees")
    _key.plot([], [], ".", color="#4a5568", label="the imputer's fill")
    _key.legend(frameon=False, fontsize=8, loc="center left")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("flow [m³/h]", color="#4a5568")
    _fig.suptitle(
        f"Gap {run_choice.value} of {len(fill_runs)}, {fill_gap_choice.value} long",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The ranking **changes with gap length**. Up to a couple of hours,
    interpolation is as good as anything, and much simpler. Beyond that,
    every method that only looks at the flow itself breaks down, because a
    pump stop can start or end inside the gap. The methods that use pressure
    stay at about 2 m³/h at every length. That floor is the pressure
    sensor's own noise: 0.02 bar at 0.01 bar per m³/h.

    Two things to keep in mind:

    - **Coverage.** An imputer that leaves a gap unfilled has coverage below
      1, and its error covers only what it did fill. Here every candidate
      filled everything.
    - **knn and iterative learn from the data they fill**, every time they
      run, while the others are fitted once. For a pipeline you will reuse,
      a fitted `regression` is easier to reason about.

    ## From the benchmark to a tiered imputer

    `Tiered` gives each gap to the method that suits its length. Read the
    boundary off the table: the longest tested gap where linear
    interpolation is within 20% of the best method.
    """)
    return


@app.cell
def _(GAPS, fs, mo, rmse):
    _close = rmse.loc["linear"] <= 1.2 * rmse.min()
    boundary = [g for g in GAPS if _close[g]][-1]
    tiered = fs.impute.Tiered(
        "flow",
        tiers=[
            {"max_gap": boundary, "method": "interpolate"},
            {
                "max_gap": None,
                "method": "regression",
                "options": {"features": ["pressure"]},
            },
        ],
    )
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"Interpolate gaps up to **{boundary}**, and fill everything longer by "
        "regression on pressure:"
    )
    return boundary, tiered


@app.cell
def _(GAPS, benchmark_imputers, candidates, clean, pd, rmse, summarize, tiered):
    _check = benchmark_imputers(
        clean,
        {"tiered": tiered, "linear": candidates["linear"]},
        streams=["flow"],
        gaps=GAPS,
        count=6,
        repeats=2,
        seed=1,
    )
    tiered_rmse = summarize(_check).loc["tiered"]
    comparison = pd.DataFrame(
        {
            "best single method": rmse.min(),
            "which": rmse.idxmin(),
            "tiered": tiered_rmse,
        }
    ).round(2)
    comparison
    return comparison, tiered_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The tiered imputer is close to the best single method at every gap
    length; no single method is. The same hidden gaps were used (the same
    `seed`), so the numbers compare directly.

    ## Using it

    In a pipeline, the tiered imputer goes after cleaning, exactly as in
    tutorial 05:

    ```python
    pipeline = fs.Pipeline([..., cleaning steps ..., tiered])
    ```

    A few habits that make the benchmark trustworthy on your own data:

    - Run it on **cleaned** data, so values are hidden only where they are
      good.
    - Test the gap lengths that **actually occur**. The flag counts after
      cleaning tell you what they are.
    - Use `repeats=3` or more before trusting small differences.
    - `pattern="random"` hides single samples instead of runs. It mimics
      scattered dropouts, but says nothing about outages.
    """)
    return


@app.cell
def _(boundary, comparison, coverage, quick_rmse, rms_error, rmse, tiered_rmse):
    # These checks run when CI runs the tutorial as a script.
    assert rms_error < 3
    assert quick_rmse.loc["linear", "30min"] < quick_rmse.loc["regression", "30min"]
    assert quick_rmse.loc["linear", "6h"] > quick_rmse.loc["regression", "6h"]
    assert (coverage == 1.0).all().all()
    # Interpolation is near the best for short gaps and far off for long ones.
    assert rmse.loc["linear", "10min"] <= 1.2 * rmse["10min"].min()
    assert rmse.loc["linear", "12h"] > 5 * rmse["12h"].min()
    # The daily cycle doesn't save the Kalman smoother from the pump stops.
    assert rmse.loc["kalman + daily cycle", "12h"] > 5
    # Using the pressure keeps the error near its noise floor at every length.
    assert (rmse.loc["regression on pressure"] < 3).all()
    assert boundary == "2h"
    assert (tiered_rmse <= 1.15 * comparison["best single method"]).all()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    Tutorial 07 hands cleaned data to a flex-pse model: lined up with the
    model's time steps, fixed into it, and prepared for fitting a surrogate.
    """)
    return


if __name__ == "__main__":
    app.run()
