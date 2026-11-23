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
    from flexparameterize import LinearRegressor

    import flexsignal as fs
    from flexsignal import handoff
    from flexsignal.integrations import flexpse as fsp
    from flexsignal.testing import synthetic

    return (
        LinearRegressor,
        build_model,
        copy,
        fs,
        fsp,
        handoff,
        mo,
        np,
        pd,
        plt,
        pyo,
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
    # 7. From raw data to a flex-pse model

    flex-signal cleans plant data; [flex-pse](https://flex-pse.github.io/flex-pse/)
    optimizes how the plant runs. This tutorial connects the two, in a plain
    script:

    1. clean a week of raw pump-flow data;
    2. line it up with a flex-pse model's hourly steps, in the plant's own
       time zone, and fix the pump's flow from it;
    3. solve the model with HiGHS;
    4. prepare the data to fit a surrogate, fit flex-pse's linear regressor,
       and score it on a held-out day.

    The plant is in California. Its historian stores UTC, as flex-signal does,
    but the flex-pse model runs on the plant's **wall clock**.

    This needs flex-pse, installed from its git tag (see the installation
    guide).
    """)
    return


@app.cell
def _(synthetic):
    # A week of 5-minute data from 6 January 2025 (UTC), with faults in the flow.
    case = (
        synthetic.plant(seed=4)
        .with_out_of_range("flow", count=3)
        .with_spikes("flow", count=6)
        .with_stuck("flow", length=24)
        .with_gaps("flow", count=4, length=2)
    )
    raw = case.signal()
    return case, raw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The core calls

    Here is the whole hand-off before we take it apart. Clean, line up with
    the model's time steps, fix, solve:
    """)
    return


@app.cell
def _(build_model, copy, fs, fsp, handoff, pyo, raw):
    SPEC = {
        "schema_version": "0.1.1",
        "name": "site",
        "elements": [
            {
                "kind": "time",
                "start_date": "2025-01-09",
                "end_date": "2025-01-10",
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

    # 1. Clean (tutorials 2 and 6 explain these steps).
    pipe = fs.Pipeline(name="flow_cleanup")
    pipe.add_step(fs.clean.Range("flow", min=0, max=500))
    pipe.add_step(fs.clean.Stuck("flow", min_duration="1h"))
    pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0))
    pipe.add_step(fs.impute.Interpolate("flow", max_gap="3h"))
    clean = fs.run(pipe.spec, raw).signal

    # 2. Line it up with the model's hourly steps, on the plant's clock.
    grid = fsp.time_grid(SPEC)
    options = handoff.GridOptions(timezone="America/Los_Angeles", gaps="error")
    values = handoff.to_grid(clean, grid, options, columns=["flow"])

    # 3. Fix the pump's flow, and solve.
    model = build_model(copy.deepcopy(SPEC))
    fsp.apply_values(model, values.to_overrides(rename={"flow": "site.pump.flow_in"}))
    result = pyo.SolverFactory("appsi_highs").solve(model)
    status = str(result.solver.termination_condition)
    status
    return SPEC, clean, grid, model, options, status, values


@app.cell
def _(grid, mo, model, pyo, values):
    _cost = pyo.value(model.costing.opex.electricity_cost)
    _units = pyo.units.get_units(model.costing.opex.electricity_cost)
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The model has **{len(grid)}** hourly steps, from {grid[0]} to {grid[-1]}
    local time. `to_grid` gave **{len(values.values["flow"])}** values in
    {values.units["flow"]}; **{values.estimated["flow"]}** of the hours rest
    on samples the cleaning filled. The solved model's electricity cost is
    **{_cost:,.2f} {_units}** (an annualized figure, as flex-pse reports it).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The model's clock

    A flex-pse model's time points are naive local times, each the start of a
    one-hour step. `time_grid` reads them from the spec (or from a built
    model). `to_grid` converts the UTC data to the facility's time zone,
    drops the zone, and averages the 5-minute samples in each step.

    The time zone matters: the same model day reads different hours of data.
    Pick one and compare it with what the model would get if the plant's
    clock were UTC:
    """)
    return


@app.cell
def _(mo):
    # Notebook control (marimo): a dropdown to choose the plant's time zone.
    zone_choice = mo.ui.dropdown(
        options=["America/Los_Angeles", "America/New_York", "Europe/Berlin"],
        value="America/Los_Angeles",
        label="plant's time zone",
    )
    zone_choice
    return (zone_choice,)


@app.cell
def _(COLORS, clean, grid, handoff, plt, style, zone_choice):
    _utc = handoff.to_grid(clean, grid, {"timezone": "UTC"}, columns=["flow"])
    _local = handoff.to_grid(
        clean, grid, {"timezone": zone_choice.value}, columns=["flow"]
    )
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.step(
        grid,
        _utc.values["flow"],
        where="post",
        color=COLORS["truth"],
        lw=2,
        label="as if the plant ran on UTC",
    )
    _ax.step(
        grid,
        _local.values["flow"],
        where="post",
        color=COLORS["signal"],
        lw=1.4,
        label=f"{zone_choice.value}",
    )
    style(_ax, "pump flow [m³/h]").set_title(
        "What the model's day reads, by time zone", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The daily peak sits in the afternoon *UTC* in the data. On a California
    plant's clock that is the morning, so the curve shifts by the zone's
    offset. Getting this wrong puts the high-flow hours against the wrong
    electricity prices.

    ## Clean first, then hand off

    `to_grid` can fill empty steps itself (`gaps="interpolate"`, `"ffill"` or
    `"zero"`), but that is a last resort: fill gaps with flex-signal, where
    each method suits the gap and every filled value is flagged, and hand off
    with `gaps="error"` so nothing is filled silently.

    `gaps="error"` only refuses *empty* steps, though. What about the raw data,
    uncleaned? Its dropouts are short, so every hour still has samples to
    average, and it is accepted:
    """)
    return


@app.cell
def _(case, clean, grid, handoff, np, options, raw):
    raw_hours = handoff.to_grid(raw, grid, options, columns=["flow"])  # accepted
    true_hours = handoff.to_grid(
        (case.clean[["flow"]], {"flow": "m^3/hr"}), grid, options, columns=["flow"]
    )
    clean_hours = handoff.to_grid(clean, grid, options, columns=["flow"])
    _truth = np.array(true_hours.values["flow"])
    raw_error = float(np.max(np.abs(np.array(raw_hours.values["flow"]) - _truth)))
    clean_error = float(np.max(np.abs(np.array(clean_hours.values["flow"]) - _truth)))
    return clean_error, clean_hours, raw_error, raw_hours, true_hours


@app.cell
def _(COLORS, clean_hours, grid, plt, raw_hours, style, true_hours):
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.step(
        grid,
        true_hours.values["flow"],
        where="post",
        color=COLORS["truth"],
        lw=3,
        label="true hourly flow",
    )
    _ax.step(
        grid,
        raw_hours.values["flow"],
        where="post",
        color=COLORS["fault"],
        lw=1.2,
        label="raw data",
    )
    _ax.step(
        grid,
        clean_hours.values["flow"],
        where="post",
        color=COLORS["signal"],
        lw=1.2,
        label="cleaned data",
    )
    style(_ax, "pump flow [m³/h]").set_title(
        "The model's day, from raw and from cleaned data", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(clean_error, mo, raw_error, values):
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    One `-999` error code falls on the model's day. It is one sample in
    twelve, but it drags that hour's mean **{raw_error:.0f} m³/h** below the
    truth, and the raw data goes into the model with that error. The cleaned
    data is within **{clean_error:.1f} m³/h** of the truth every hour, and
    `values.estimated` says **{values.estimated["flow"]}** of its hours rest
    on samples the cleaning filled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Into the model

    `values.to_overrides(rename=...)` names each column by its model path and
    attaches its units. `fsp.apply_values` takes either a spec or a built
    model, with the same result:

    - `fsp.apply_values(SPEC, overrides)` returns a copy of the spec with
      `fix` elements, to build (and sets a flat price:
      `"costing.energy_prices.electrical"`);
    - `fsp.apply_values(model, overrides)` fixes a built model's variables in
      place, checked against their bounds (and sets a price the model holds
      as a mutable Param).

    The core cell above applied them to the built model. The pump's power follows from its
    fixed flow at 0.4 kWh per m³:
    """)
    return


@app.cell
def _(COLORS, grid, model, np, plt, pyo, style, values):
    pump_flow = np.array(
        [pyo.value(model.site.pump.flow_in[t]) for t in model.site.pump.flow_in]
    )
    pump_power = np.array(
        [
            pyo.value(model.site.pump.power_electrical[t])
            for t in model.site.pump.power_electrical
        ]
    )
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.step(
        grid,
        values.values["flow"],
        where="post",
        color=COLORS["signal"],
        lw=1.4,
        label="flow fixed from data [m³/h]",
    )
    _ax.step(
        grid,
        pump_power,
        where="post",
        color=COLORS["filled"],
        lw=1.4,
        label="pump power [kW]",
    )
    style(_ax).set_title("The solved model", loc="left", fontsize=10)
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return pump_flow, pump_power


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit data for a surrogate

    flex-pse models often describe a unit with a fitted surrogate, such as
    pressure or power from flow. `handoff.fit_data` prepares the data: it
    joins the inputs and the one output on time, resamples, applies a gap
    rule, logs each step, and hashes the result. flex-signal stops there; you
    fit flex-pse's regressor yourself.

    Here the synthetic plant's pressure follows its flow
    (4 + 0.01 × flow bar). Fit it on the first six days, hourly, and score it
    on the last:
    """)
    return


@app.cell
def _(LinearRegressor, clean, fs, handoff):
    fit = handoff.fit_data(
        clean.select(["flow"]),
        clean.select(["pressure"]),
        {"resample": {"freq": "1h"}, "gaps": "drop"},
    )
    X_train, y_train, X_test, y_test = handoff.split_holdout(fit.X, fit.y, "1D")
    regressor = LinearRegressor().fit(
        X_train, y_train, input_units=fit.input_units, output_units=fit.output_units
    )
    coefficients = regressor.to_fit_result().coefficients
    predicted = coefficients["intercept"] + coefficients["flow"] * X_test["flow"]
    scores = fs.eval.metrics(y_test["pressure"], predicted)
    # regressor.to_surrogate_spec() gives the surrogate for a flex-pse model.
    {"coefficients": coefficients, "held-out scores": scores}
    return X_test, X_train, coefficients, fit, predicted, scores, y_test, y_train


@app.cell
def _(COLORS, X_test, X_train, coefficients, np, plt, scores, style, y_test, y_train):
    _flow = np.linspace(
        min(X_train["flow"].min(), X_test["flow"].min()),
        max(X_train["flow"].max(), X_test["flow"].max()),
        50,
    )
    _line = coefficients["intercept"] + coefficients["flow"] * _flow
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True)
    _panels = [
        (
            _axes[0],
            X_train,
            y_train,
            "measured (fitted on)",
            f"Fitted on: {len(X_train)} hours",
        ),
        (
            _axes[1],
            X_test,
            y_test,
            "measured (held out)",
            f"The held-out day: RMSE {scores['rmse']:.3f} bar",
        ),
    ]
    for _ax, _X, _y, _label, _title in _panels:
        _ax.plot(
            _X["flow"], _y["pressure"], ".", ms=4, color=COLORS["signal"], label=_label
        )
        _ax.plot(_flow, _line, color=COLORS["fault"], lw=1.4, label="fitted surrogate")
        _ax.set_xlabel("flow [m³/h]", color="#4a5568")
        style(_ax, "pressure [bar]").set_title(_title, loc="left", fontsize=10)
        _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `fit.steps` records what was done to the data, and `fit.data_hash`
    identifies it exactly: the same data gives the same hash on every run.

    ## In flex-orchestrator

    flex-orchestrator runs this hand-off at scale, in studies and sweeps, by
    calling the same functions: its `model_inputs` use `to_grid` with the
    same options, and its `fit_surrogate` uses `fit_data`. A spec that works
    in this script works there too.
    """)
    return


@app.cell
def _(
    clean,
    coefficients,
    fit,
    np,
    pump_flow,
    pump_power,
    clean_error,
    raw_error,
    scores,
    status,
    values,
):
    # These checks run when CI runs the tutorial as a script.
    assert status == "optimal"
    assert len(values.values["flow"]) == 24
    assert np.allclose(pump_flow, values.values["flow"])
    assert np.allclose(pump_power, 0.4 * pump_flow)
    assert raw_error > 50  # the raw data is accepted, with a -999 in one hour
    assert clean_error < 3
    assert clean.flag_counts()["flow"]["flags"]["IMPUTED_SHORT"] > 0
    assert abs(coefficients["flow"] - 0.01) < 0.002
    assert abs(coefficients["intercept"] - 4.0) < 0.2
    assert scores["rmse"] < 0.05
    assert fit.steps[0]["step"] == "align"
    return


if __name__ == "__main__":
    app.run()
