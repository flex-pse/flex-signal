import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    import tempfile
    from pathlib import Path

    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" are not flex-signal; in your own
    # script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    import flexsignal as fs
    from flexsignal.eval import benchmark_imputers, summarize
    from flexsignal.testing import synthetic

    return (
        LinearRegression,
        Path,
        benchmark_imputers,
        fs,
        json,
        mo,
        np,
        pd,
        plt,
        summarize,
        synthetic,
        sys,
        tempfile,
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
    # 10. Your own methods

    The built-in steps cover the common cases, but a plant often needs a method
    of its own: a despiking rule tuned to one meter, an imputer that knows the
    plant's daily routine, a regression model a team trusts. This tutorial
    adds three of them without changing flex-signal:

    1. register a **cleaner**, an **imputer** and a **model** from plain Python;
    2. score them against the injected faults and against built-in methods;
    3. run them from a JSON spec, and save and reload the fitted pipeline;
    4. see what is refused: unknown methods, bad options, import paths;
    5. package them so `pip install` makes them available.

    The rule behind it: **only registered methods run**. A spec names a method,
    never code to import, so a spec from someone else can only use what you
    chose to register.
    """)
    return


@app.cell
def _(synthetic):
    # Two weeks of 5-minute data from 6 January 2025 (UTC), with spikes and a
    # stuck stretch in the flow.
    case = (
        synthetic.plant(seed=5, periods=14 * 288)
        .with_spikes("flow", count=15)
        .with_stuck("flow", count=1)
    )
    raw = case.signal()
    spikes = case.fault_mask("spike", "flow")
    return case, raw, spikes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A cleaner

    A cleaner is a function that takes one stream, as a pandas Series, and
    returns True for each value to reject. Its keyword-only parameters, after
    the `*`, become the step's options, checked by their type hints. The
    docstring's `Args:` lines become the options' descriptions.
    `register_cleaner` does the rest: rejected values become NaN and gain the
    flag, and values that are already missing are left alone.
    """)
    return


@app.cell
def _(fs, pd):
    def despike(values: pd.Series, *, window: str = "2h", k: float = 8.0) -> pd.Series:
        """Reject values far from a rolling median.

        Args:
            values: The stream.
            window: Width of the rolling window, e.g. 2h.
            k: How many robust standard deviations from the median count as a
                spike.
        """
        median = values.rolling(window, center=True, min_periods=1).median()
        deviation = (values - median).abs()
        spread = 1.4826 * deviation.rolling(window, center=True, min_periods=1).median()
        return deviation > k * spread.clip(lower=1e-9)

    fs.register_cleaner("PlantDespike", despike, flag="OUTLIER")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It is now `fs.clean.PlantDespike`, used like any built-in step. The
    synthetic data knows where the spikes are, so we can count what it caught,
    and compare it with the built-in Hampel filter on the same data:
    """)
    return


@app.cell
def _(fs, pd, raw, spikes):
    def caught(step):
        """Run one cleaner; count the spikes it caught and its false alarms."""
        out = fs.run(fs.Pipeline([step]).spec, raw).signal
        rejected = out.has_flag(fs.Flag.OUTLIER)["flow"]
        return {
            "spikes caught": int((rejected & spikes).sum()),
            "false alarms": int((rejected & ~spikes).sum()),
        }

    cleaning = pd.DataFrame(
        {
            "PlantDespike (ours)": caught(fs.clean.PlantDespike("flow")),
            "hampel (built in)": caught(
                fs.clean.Hampel("flow", window="1h", n_sigma=6)
            ),
        }
    ).T
    cleaning
    return (cleaning,)


@app.cell
def _(COLORS, fs, pd, plt, raw, spikes, style):
    _out = fs.run(fs.Pipeline([fs.clean.PlantDespike("flow")]).spec, raw).signal
    _rejected = _out.has_flag(fs.Flag.OUTLIER)["flow"]
    _day = slice(raw.index[0], raw.index[0] + pd.Timedelta("3D"))
    _flow = raw.values["flow"][_day]
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.plot(_flow.index, _flow, color=COLORS["truth"], lw=0.8, label="raw flow")
    _hit = _rejected[_day] & spikes[_day]
    _ax.scatter(
        _flow.index[_hit],
        _flow[_hit],
        color=COLORS["fault"],
        s=18,
        zorder=3,
        label="rejected by PlantDespike",
    )
    style(_ax, "flow [m³/h]").set_title(
        "Three days: every injected spike rejected", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(cleaning, mo):
    mo.md(f"""
    Both catch all {cleaning.iloc[0, 0]} spikes; ours raises
    {cleaning.iloc[0, 1]} false alarms to Hampel's {cleaning.iloc[1, 1]}. That
    is the point of scoring it: a method of your own goes through the same
    checks as a built-in one, so you can tell whether it earns its place.

    ## An imputer and a model

    An imputer gets every stream (the one being filled first) and a boolean
    Series of the gaps it may fill, and returns proposed values; only the gaps
    are used, and they are flagged `IMPUTED_MODEL`. `max_gap` is added by
    flex-signal. This one fills a gap with the same time yesterday, shifted so
    it meets the values either side of the gap.

    A model is a regressor class that works like scikit-learn's, described by
    its fitted `coef_` and `intercept_`. Registered, it is a `model` choice for
    `calibrate` and `regression`, and only its coefficients are kept, as JSON.
    """)
    return


@app.cell
def _(LinearRegression, fs, np, pd):
    def same_time_yesterday(
        values: pd.DataFrame, gaps: pd.Series, *, period: str = "1D"
    ) -> np.ndarray:
        """Fill a gap with the same time one period earlier, shifted to fit.

        Args:
            values: The streams, the one being filled first.
            gaps: Where to fill.
            period: How far back to look, e.g. 1D for the same time yesterday.
        """
        y = values.iloc[:, 0]
        lag = int(pd.Timedelta(period) / (y.index[1] - y.index[0]))
        earlier = y.shift(lag).to_numpy()
        now = y.to_numpy()
        gap = gaps.to_numpy()
        out = np.full(len(y), np.nan)
        starts = np.flatnonzero(gap & ~np.r_[False, gap[:-1]])
        ends = np.flatnonzero(gap & ~np.r_[gap[1:], False])
        for first, last in zip(starts, ends, strict=True):
            before, after = first - 1, last + 1
            if before < 0 or after >= len(y):
                continue
            offsets = now[[before, after]] - earlier[[before, after]]
            if np.isnan(offsets).any():
                continue
            share = (np.arange(first, last + 1) - before) / (after - before)
            shift = offsets[0] + (offsets[1] - offsets[0]) * share
            out[first : last + 1] = earlier[first : last + 1] + shift
        return out

    class PositiveLeastSquares(LinearRegression):
        """Least squares with every coefficient at least zero."""

        def __init__(self):
            super().__init__(positive=True)

    fs.register_imputer("SameTimeYesterday", same_time_yesterday)
    fs.register_model(
        "PositiveLeastSquares",
        PositiveLeastSquares,
        description="Least squares, every coefficient at least zero.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scoring the imputers

    `benchmark_imputers` hides stretches of the cleaned flow, lets each imputer
    fill them, and scores the result against what was hidden. Our two methods
    compete with linear interpolation and the Kalman smoother, on gaps of half
    an hour to six hours:
    """)
    return


@app.cell
def _(benchmark_imputers, fs, raw, summarize):
    cleaned = fs.run(fs.Pipeline([fs.clean.PlantDespike("flow")]).spec, raw).signal
    DAILY = [{"period": "1D", "harmonics": 3}]
    benchmark = benchmark_imputers(
        cleaned,
        {
            "SameTimeYesterday (ours)": fs.impute.SameTimeYesterday(
                "flow", max_gap="1D"
            ),
            "regression with PositiveLeastSquares (ours)": fs.impute.Regression(
                "flow",
                features=["pressure"],
                fourier=DAILY,
                model="PositiveLeastSquares",
            ),
            "linear (built in)": fs.impute.Interpolate("flow", max_gap="1D"),
            "kalman (built in)": fs.impute.Kalman("flow", max_gap="1D", seasonal=DAILY),
        },
        streams=["flow"],
        gaps=["30min", "2h", "6h"],
        count=8,
    )
    rmse = summarize(benchmark)
    rmse.round(2)
    return DAILY, rmse


@app.cell
def _(COLORS, np, plt, rmse, style):
    _colors = [COLORS["filled"], "#805ad5", COLORS["truth"], COLORS["signal"]]
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _width = 0.2
    _x = np.arange(len(rmse.columns))
    for _i, (_name, _row) in enumerate(rmse.iterrows()):
        _ax.bar(_x + (_i - 1.5) * _width, _row, _width, color=_colors[_i], label=_name)
    _ax.set_xticks(_x, [f"{g} gaps" for g in rmse.columns])
    style(_ax, "RMSE [m³/h]").set_title(
        "Hidden values filled back: lower is better", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, rmse):
    _ours = rmse.loc["SameTimeYesterday (ours)", "6h"]
    _linear = rmse.loc["linear (built in)", "6h"]
    _kalman = rmse.loc["kalman (built in)", "6h"]
    _regression = rmse.loc["regression with PositiveLeastSquares (ours)", "6h"]
    mo.md(f"""
    On six-hour gaps, the same time yesterday (RMSE {_ours:.2f} m³/h) beats a
    straight line ({_linear:.2f}), because it follows the daily cycle, but the
    Kalman smoother, which models that cycle, does better still ({_kalman:.2f}).
    The regression on pressure, fitted with our model, has about the same
    error ({_regression:.2f}) whatever the gap's length: it doesn't use the
    values either side of a gap at all.

    ## The same methods from a JSON spec

    Once registered, a method is used by name, so a JSON spec can use it like
    any built-in step. The spec names nothing to import: `requires` only tells
    someone who doesn't have the methods which package to install.
    """)
    return


@app.cell
def _(DAILY, fs, json, raw):
    SPEC = {
        "schema_version": "0.1.0",
        "name": "my_methods",
        "requires": ["myplant.methods"],
        "steps": [
            {"kind": "PlantDespike", "streams": ["flow"]},
            {"kind": "interpolate", "streams": ["flow"], "max_gap": "15min"},
            {"kind": "SameTimeYesterday", "streams": ["flow"], "max_gap": "1D"},
        ],
    }
    python_pipe = fs.Pipeline(
        [
            fs.clean.PlantDespike("flow"),
            fs.impute.Interpolate("flow", max_gap="15min"),
            fs.impute.SameTimeYesterday("flow", max_gap="1D"),
        ],
        name="my_methods",
    )
    from_json = fs.run(json.dumps(SPEC), raw)
    from_python = fs.run(python_pipe.spec, raw)
    same = from_json.data.equals(from_python.data) and from_json.flags.equals(
        from_python.flags
    )
    provenance = from_json.provenance[2]["function"]
    _ = DAILY
    {"same output": same, "history names": provenance}
    return SPEC, from_json, same


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each history record names the function behind a registered step: its
    module, its name, and the version of the package that provides it. So a
    result says which code produced it.

    ## Saving and reloading

    The fitted state of every step is JSON, so the pipeline saves like any
    other. Loading it needs the same methods registered first:
    """)
    return


@app.cell
def _(Path, from_json, fs, raw, tempfile):
    folder = Path(tempfile.mkdtemp()) / "my_methods"
    from_json.fitted.save(folder)
    reloaded = fs.FittedPipeline.load(folder)
    reloaded_matches = reloaded.transform(raw).values.equals(from_json.data)
    sorted(p.name for p in (folder / "fitted").iterdir())
    return (reloaded_matches,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is refused

    A spec that names a method nobody registered fails when it loads, before
    any data is touched, and says what to do. An option of the wrong type fails
    too, naming the field. And a spec can't name code to import, whether as a
    step's kind or as a model:
    """)
    return


@app.cell
def _(SPEC, fs):
    def refusal(spec):
        """Load a spec; return the field and message it is refused with."""
        try:
            fs.load_spec(spec)
        except fs.SignalConfigError as exc:
            return {"field": exc.field, "message": exc.message[:150] + " ..."}
        return {"field": None, "message": "loaded"}

    _step = SPEC["steps"][0]
    refusals = {
        "unregistered kind": refusal(
            {**SPEC, "steps": [{**_step, "kind": "OtherDespike"}]}
        ),
        "bad option": refusal({**SPEC, "steps": [{**_step, "k": "large"}]}),
        "an import path": refusal(
            {**SPEC, "steps": [{**_step, "kind": "myplant.methods:despike"}]}
        ),
        "a class as model": refusal(
            {
                **SPEC,
                "steps": [
                    {
                        "kind": "regression",
                        "streams": ["flow"],
                        "features": ["pressure"],
                        "model": "sklearn.linear_model:Lasso",
                    }
                ],
            }
        ),
    }
    refusals
    return (refusals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packaging the methods

    To share the methods, put the functions and the `fs.register_*` calls in a
    module of a package, say `myplant/methods.py`, and expose it through the
    `flexsignal.steps` entry-point group in the package's `pyproject.toml`:

    ```toml
    [project.entry-points."flexsignal.steps"]
    myplant = "myplant.methods"
    ```

    Once the package is `pip install`ed, its methods are available wherever
    flex-signal runs, including Ray workers. Without an entry point, the caller
    loads the module by name: `fs.load_plugins("myplant.methods")`, or
    `fs.run(spec, data, plugins=["myplant.methods"])`. Here is that route, with
    a small module written to a temporary folder:
    """)
    return


@app.cell
def _(Path, fs, sys, tempfile):
    _package = Path(tempfile.mkdtemp())
    (_package / "myplant_methods.py").write_text(
        '''
import flexsignal as fs


def clip_negative(values, *, floor: float = 0.0):
    """Reject values below a floor."""
    return values < floor


fs.register_cleaner("MyplantNegative", clip_negative, flag="OUT_OF_RANGE")
''',
        encoding="utf-8",
    )
    sys.path.insert(0, str(_package))
    loaded = fs.load_plugins("myplant_methods")
    loaded
    return (loaded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    - The explanation page *Your own methods* sets out the rules, and the
      how-to guide *Add your own method* adds a forecaster too.
    - flex-signal's own adapters to optional libraries (PyPOTS, Chronos, pyod,
      darts, sktime) are registered through the same functions; tutorial 9
      uses some of them.
    """)
    return


@app.cell
def _(cleaning, loaded, refusals, reloaded_matches, rmse, same):
    # The tutorial's claims, checked.
    assert cleaning.loc["PlantDespike (ours)", "spikes caught"] == 15
    assert cleaning.loc["PlantDespike (ours)", "false alarms"] <= 5
    assert (
        rmse.loc["SameTimeYesterday (ours)", "6h"] < rmse.loc["linear (built in)", "6h"]
    )
    assert same and reloaded_matches
    assert refusals["unregistered kind"]["field"] == "steps.0.kind"
    assert "register it first" in refusals["unregistered kind"]["message"]
    assert refusals["bad option"]["field"] == "steps.0.k"
    assert refusals["an import path"]["field"] == "steps.0.kind"
    assert refusals["a class as model"]["field"] == "steps.0.model"
    assert loaded == ["MyplantNegative"]
    return


if __name__ == "__main__":
    app.run()
