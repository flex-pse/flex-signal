import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy

    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from flexops import build_model

    import flexsignal as fs
    from flexsignal import handoff
    from flexsignal.integrations import flexpse as fsp
    from flexsignal.testing import synthetic

    return build_model, copy, fs, fsp, handoff, mo, np, pd, plt, pyo, synthetic


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
    METHOD_COLORS = {
        "forecast_persistence": "#805ad5",
        "forecast_seasonal_naive": COLORS["fault"],
        "forecast_structural": COLORS["signal"],
        "forecast_regression": COLORS["filled"],
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
    return COLORS, METHOD_COLORS, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. Forecasting and backtesting

    To plan tomorrow's operation, a flex-pse model needs tomorrow's flow
    before anyone has measured it. This tutorial forecasts it:

    1. clean three weeks of pump-flow data and forecast the next day;
    2. compare two baselines with two models, and their intervals;
    3. score every method fairly with a **backtest** that can't see the
       future it is scored on;
    4. fix tomorrow's forecast into a flex-pse model and solve it.

    The plant's flow has a daily cycle (busy afternoons) and a weekly one. The
    synthetic data knows the true flow, so every forecast can be checked
    against it. The last part needs flex-pse, installed from its git tag (see
    the installation guide).
    """)
    return


@app.cell
def _(synthetic):
    # Three weeks of 5-minute data from 6 January 2025 (UTC), with spikes and
    # gaps in the flow.
    DAYS = 21
    case = (
        synthetic.plant(seed=7, periods=DAYS * 288)
        .with_spikes("flow", count=12)
        .with_gaps("flow", count=6, length=4)
    )
    raw = case.signal()
    # The true hourly flow, one day longer than the data: tomorrow included.
    truth = (
        synthetic.plant(seed=7, periods=(DAYS + 1) * 288)
        .underlying["flow"]
        .resample("1h")
        .mean()
    )
    return raw, truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The core calls

    A forecast is a pipeline step, so it goes after the cleaning. It needs a
    regular index; hourly means suit a day-ahead forecast anyway. Then
    `fs.backtest` scores it against a baseline on the data's past:
    """)
    return


@app.cell
def _(fs, raw):
    SEASONS = [{"period": "1D", "harmonics": 2}, {"period": "7D", "harmonics": 1}]

    # 1. Clean, resample to hours, forecast one day ahead.
    pipe = fs.Pipeline(name="flow_forecast")
    pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6))
    pipe.add_step(fs.select.Resample(freq="1h"))
    pipe.add_step(
        fs.forecast.Structural(
            "flow", horizon="1D", seasonal=SEASONS, quantiles=[0.1, 0.9]
        )
    )
    forecast = fs.run(pipe.spec, raw).signal

    # 2. Backtest it against the same time yesterday, on four past days.
    backtest = fs.backtest(
        pipe.spec,
        raw,
        folds=4,
        compare=[fs.forecast.SeasonalNaive("flow", horizon="1D", season="1D")],
    )
    backtest.summary()
    return SEASONS, backtest, forecast, pipe


@app.cell
def _(backtest, forecast, fs, mo):
    _ahead = forecast.has_flag(fs.Flag.FORECAST)["flow"]
    _skill = backtest.summary().set_index("method").loc["forecast_structural", "skill"]
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The forecast was issued at **{forecast.meta["flow"].issue_time}**, the last
    hour with data, and fills the **{int(_ahead.sum())}** hours after it,
    flagged `FORECAST`. `flow__q10` and `flow__q90` hold its 80% interval. In
    the backtest, its skill against seasonal-naive is **{_skill:.2f}**: its
    RMSE is {1 - _skill:.0%} of the baseline's.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baselines first

    A forecaster is only worth its complexity if it beats a simple rule.
    Two are built in:

    - `Persistence`: the last value, held. Fine for the next few minutes,
      poor for a day with a strong daily cycle.
    - `SeasonalNaive`: the value one season earlier, here the same hour
      yesterday. It follows the daily cycle, but carries yesterday's noise
      and misses the weekly drift.

    The models:

    - `Structural`: statsmodels' structural time-series model, a level plus
      daily and weekly cycles, fitted by maximum likelihood.
    - `Regression`: a linear model of the same cycles, as sine and cosine
      terms.

    Each step below forecasts the last day of the data, so it can be checked
    against what happened. Choose one to see its interval:
    """)
    return


@app.cell
def _(SEASONS, fs, pipe, raw):
    # The same cleaning, each method in turn, issued a day before the data ends.
    ISSUE = "2025-01-25T23:00:00Z"
    METHODS = {
        "forecast_persistence": fs.forecast.Persistence(
            "flow", horizon="1D", quantiles=[0.1, 0.9]
        ),
        "forecast_seasonal_naive": fs.forecast.SeasonalNaive(
            "flow", horizon="1D", quantiles=[0.1, 0.9]
        ),
        "forecast_structural": fs.forecast.Structural(
            "flow", horizon="1D", seasonal=SEASONS, quantiles=[0.1, 0.9]
        ),
        "forecast_regression": fs.forecast.Regression(
            "flow", horizon="1D", fourier=SEASONS, quantiles=[0.1, 0.9]
        ),
    }
    cleaning = list(pipe.spec.steps[:-1])
    last_day = {
        kind: fs.run(
            fs.Pipeline(
                [*cleaning, step.model_copy(update={"issue_time": ISSUE})], name=kind
            ).spec,
            raw,
        ).signal
        for kind, step in METHODS.items()
    }
    return METHODS, last_day


@app.cell
def _(METHODS, mo):
    # Notebook control: which method's interval to draw.
    shown = mo.ui.dropdown(
        options=list(METHODS), value="forecast_seasonal_naive", label="Interval of"
    )
    shown
    return (shown,)


@app.cell
def _(COLORS, METHOD_COLORS, fs, last_day, np, plt, shown, style, truth):
    _fig, _ax = plt.subplots(figsize=(9, 3.4))
    _history = truth.loc["2025-01-23":"2025-01-25"]
    _ax.plot(_history.index, _history, color=COLORS["truth"], lw=1.2, label="true flow")
    errors = {}
    for _kind, _sf in last_day.items():
        _rows = _sf.has_flag(fs.Flag.FORECAST)["flow"]
        _pred = _sf.values["flow"][_rows]
        _actual = truth.reindex(_pred.index)
        errors[_kind] = float(np.sqrt(np.mean((_pred - _actual) ** 2)))
        if _kind == shown.value:
            _ax.fill_between(
                _pred.index,
                _sf.values["flow__q10"][_rows],
                _sf.values["flow__q90"][_rows],
                color=METHOD_COLORS[_kind],
                alpha=0.15,
                lw=0,
                label="80% interval",
            )
        _ax.plot(
            _pred.index,
            _pred,
            color=METHOD_COLORS[_kind],
            lw=1.6 if _kind == shown.value else 1.0,
            label=f"{_kind.removeprefix('forecast_')} (RMSE {errors[_kind]:.2f})",
        )
    _truth_day = truth.loc["2025-01-26"]
    _ax.plot(_truth_day.index, _truth_day, color=COLORS["truth"], lw=1.2, ls="--")
    _ax.axvline(_truth_day.index[0], color="#a0aec0", lw=0.8)
    style(_ax, "flow [m³/h]").set_title(
        "Forecasting the last day of the data", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _fig.tight_layout()
    _fig
    return (errors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Persistence is far off by the afternoon, and seasonal-naive repeats
    yesterday's noise and misses the weekly drift. Both models follow the
    cycles almost exactly, with intervals too narrow to see (pick one above).
    That is the synthetic data: its flow is an exact daily and weekly cycle
    plus sensor noise, scored here against the noise-free truth. A real plant
    varies from day to day, so real forecasts miss by more, and the gap to
    the baseline is smaller.

    The intervals differ in kind: `Structural`'s come from the model and
    widen as the forecast reaches further ahead; `Regression`'s are the
    spread of its training errors, the same at every hour.

    ## A fair score: backtesting

    One day is one sample. A **backtest** issues the forecast on several past
    days and scores each one. For each fold, it hides every stream after the
    issue time, then fits and runs the whole pipeline, cleaning included. A
    centred cleaning window or an imputer can't pull information back from
    the future, because the future isn't there.

    `compare` swaps other forecast steps into the same pipeline, so every
    method is scored on exactly the same cleaned data. The first one is the
    baseline that `skill` is measured against:
    """)
    return


@app.cell
def _(METHODS, fs, pipe, raw):
    scores = fs.backtest(
        pipe.spec,
        raw,
        folds=5,
        compare=[
            METHODS["forecast_seasonal_naive"],
            METHODS["forecast_persistence"],
            METHODS["forecast_regression"],
        ],
    )
    summary = scores.summary().round(3)
    summary
    return scores, summary


@app.cell
def _(COLORS, METHOD_COLORS, plt, scores, style, truth):
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _span = truth.loc["2025-01-21":"2025-01-26"]
    _ax.plot(_span.index, _span, color=COLORS["truth"], lw=1.2, label="true flow")
    _forecasts = scores.forecasts
    for _kind in ("forecast_seasonal_naive", "forecast_structural"):
        for _fold, _rows in _forecasts[_forecasts["method"] == _kind].groupby("fold"):
            _ax.plot(
                _rows["time"],
                _rows["forecast"],
                color=METHOD_COLORS[_kind],
                lw=1.1,
                label=_kind.removeprefix("forecast_") if _fold == 0 else None,
            )
    for _issue in _forecasts["issue_time"].unique():
        _ax.axvline(_issue, color="#cbd5e0", lw=0.8)
    style(_ax, "flow [m³/h]").set_title(
        "Five backtest folds, each issued at a grey line", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `scores.scores` has one row per method and fold, `scores.forecasts` every
    forecast value with its actual value and lead time, and
    `scores.to_dict()` is JSON for a report or a flex-orchestrator ledger.
    `coverage` is the share of actual values inside the 80% interval; near
    0.8 means the interval is honest.

    ## Tomorrow, into a flex-pse model

    A forecast is a prediction, so it reaches a model the way measured data
    does: `to_grid` lines it up with the model's hourly steps, and
    `apply_values` fixes the pump's flow to it. The model here is tomorrow,
    27 January, the day after the data ends:
    """)
    return


@app.cell
def _(build_model, copy, forecast, fsp, handoff, pyo):
    SPEC = {
        "schema_version": "0.1.1",
        "name": "site",
        "elements": [
            {
                "kind": "time",
                "start_date": "2025-01-27",
                "end_date": "2025-01-28",
                "time_step": "1 hr",
            },
            {
                "kind": "property_package",
                "name": "properties",
                "property_class": "SimpleAqueousFlow",
            },
            {
                "kind": "costing",
                "name": "costing",
                "energy_prices": {"electrical": {"value": 0.15, "units": "USD/kWh"}},
            },
            {"kind": "plant", "name": "site"},
            {
                "kind": "unit",
                "name": "site.tank",
                "unit_model_class": "Tank",
                "costing_package": None,
                "construction_options": {
                    "max_volume": {"value": 2000.0, "units": "m^3"},
                    "initial_volume": {"value": 1000.0, "units": "m^3"},
                },
                "connections": [
                    {"port": "outlet", "to": "site.pump.inlet", "name": "arc_0"}
                ],
            },
            {
                "kind": "unit",
                "name": "site.pump",
                "unit_model_class": "ConstantEnergyIntensityModel",
                "construction_options": {
                    "energy_intensity": {"value": 0.4, "units": "kWh/m^3"}
                },
            },
            {"kind": "objective", "name": "objective"},
        ],
    }
    # The plant's model runs on UTC here, so tomorrow's 24 hours are exactly the
    # forecast's 24 hours. Tutorial 7 shows a plant on local time.
    values = handoff.to_grid(
        forecast,
        fsp.time_grid(SPEC),
        handoff.GridOptions(timezone="UTC"),
        columns=["flow"],
    )
    overrides = values.to_overrides(rename={"flow": "site.pump.flow_in"})
    model = build_model(fsp.apply_values(copy.deepcopy(SPEC), overrides))
    status = str(
        pyo.SolverFactory("appsi_highs").solve(model).solver.termination_condition
    )
    status
    return model, status, values


@app.cell
def _(COLORS, forecast, fs, model, np, plt, pyo, style, truth, values):
    pump_flow = np.array(
        [pyo.value(model.site.pump.flow_in[t]) for t in model.site.pump.flow_in]
    )
    _ahead = forecast.has_flag(fs.Flag.FORECAST)["flow"]
    _times = forecast.index[_ahead]
    tomorrow_error = float(np.abs(pump_flow - truth.reindex(_times).to_numpy()).max())
    _mean = forecast.values["flow"][_ahead]
    _low = forecast.values["flow__q10"][_ahead] - _mean
    _high = forecast.values["flow__q90"][_ahead] - _mean
    _fig, (_ax, _bx) = plt.subplots(
        2,
        1,
        figsize=(9, 4.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.4]},
    )
    _ax.step(
        _times,
        pump_flow,
        where="post",
        color=COLORS["signal"],
        lw=1.4,
        label="pump flow fixed in the model",
    )
    _ax.plot(
        _times,
        truth.reindex(_times),
        color=COLORS["truth"],
        lw=1.2,
        ls="--",
        label="what will happen",
    )
    style(_ax, "flow [m³/h]").set_title(
        f"Tomorrow in the model: {values.estimated['flow']} hours rest on the forecast",
        loc="left",
        fontsize=10,
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    # The interval is under 1 m³/h wide, too thin to see on the flow's scale, so
    # it gets its own panel: everything relative to the forecast.
    _bx.fill_between(
        _times,
        _low,
        _high,
        step="post",
        color=COLORS["signal"],
        alpha=0.2,
        lw=0,
        label=f"80% interval (±{float(_high.mean()):.1f} m³/h)",
    )
    _bx.axhline(0, color=COLORS["signal"], lw=1)
    _bx.plot(
        _times,
        truth.reindex(_times).to_numpy() - _mean.to_numpy(),
        color=COLORS["truth"],
        lw=1.2,
        ls="--",
        label="what will happen",
    )
    style(_bx, "vs forecast [m³/h]")
    _bx.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _fig.tight_layout()
    _fig
    return pump_flow, tomorrow_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The lower panel shows the forecast's 80% interval, from `flow__q10` to
    `flow__q90`: the model expects the actual flow to fall inside it 8 hours in
    10. It is under 1 m³/h wide, too thin to see on the flow's own scale, so it
    is drawn relative to the forecast.

    `values.estimated` counts the model's steps that rest on forecast (or
    filled, or corrected) values, so a result built on a forecast says so.

    ## What's next

    - The forecast steps take `exog`: streams known ahead, such as a price
      schedule or a weather forecast, as rows after the issue time. A backtest
      keeps them visible. See the how-to guide *Forecast and backtest*.
    - `Regression` can also use the target's own recent values (`lags`) and
      streams known only up to now (`past`, at a lag of at least the horizon).
    - Save the fitted pipeline and run `transform` on tomorrow's data: it
      forecasts from the newest hour without refitting.
    """)
    return


@app.cell
def _(errors, forecast, fs, scores, status, summary, tomorrow_error, values):
    # These checks run when CI runs the tutorial as a script.
    _skill = summary.set_index("method")["skill"]
    assert _skill["forecast_structural"] > 0.5
    assert _skill["forecast_regression"] > 0.5
    assert _skill["forecast_persistence"] < 0
    assert errors["forecast_structural"] < errors["forecast_seasonal_naive"] / 2
    assert (scores.scores["n"] == 24).all()
    assert int(forecast.has_flag(fs.Flag.FORECAST)["flow"].sum()) == 24
    assert forecast.meta["flow"].issue_time == "2025-01-26T23:00:00+00:00"
    assert status == "optimal"
    assert values.estimated["flow"] == 24
    assert tomorrow_error < 3
    return


if __name__ == "__main__":
    app.run()
