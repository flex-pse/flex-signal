import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # This tutorial needs flex-pse with the ARIMA surrogate (flex-pse main
    # today, the next release after 0.3.1):
    #   pip install "flex-pse[parameterize,solvers] @ git+https://github.com/flex-pse/flex-pse.git@main"
    # needs-module: flexparameterize.regression.arima
    # needs-module: flexops
    # docs-page: prerendered
    #
    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" are not flex-signal; in your own
    # script, print the result or leave them out.
    import copy
    import json
    import tempfile
    from pathlib import Path

    import eeco.costs as eeco_costs
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyomo.environ as pyo
    from flexops import build_model
    from flexops.costing import load_tariff, price_series
    from flexops.surrogates import ArimaSurrogate
    from flexparameterize.regression.arima import ArimaRegressor

    import flexsignal as fs
    from flexsignal import handoff
    from flexsignal.integrations import flexpse as fsp
    from flexsignal.testing import synthetic

    return (
        ArimaRegressor,
        ArimaSurrogate,
        Path,
        build_model,
        copy,
        eeco_costs,
        fs,
        fsp,
        handoff,
        json,
        load_tariff,
        mo,
        np,
        pd,
        plt,
        price_series,
        pyo,
        synthetic,
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
    # One colour and line style per case, used in every panel of the last figure.
    CASE_LOOKS = [
        {"color": "#4a5568", "ls": "-", "lw": 1.6},  # baseline
        {"color": COLORS["signal"], "ls": "-", "lw": 1.6},  # scheduled, measured
        {"color": COLORS["fault"], "ls": "--", "lw": 1.4},  # scheduled, forecast
    ]

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
    return CASE_LOOKS, COLORS, style


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 12. Scheduling a biogas digester

    A biogas plant feeds sludge to a digester, burns the gas in an engine (a
    CHP unit) and sells the electricity. Electricity is worth much more in
    the evening, and the gas follows the feed within hours. So if a tank
    holds the incoming feed back, the plant can digest more of it before the
    evening and sell more power when it pays most.

    This tutorial plans one day of that, from raw plant data to a solved
    flex-pse model:

    1. look at the digester's data: feed, an online gas meter and a lab check;
    2. clean it and put it on 15-minute steps;
    3. correct the meter against the lab;
    4. fit an ARIMA surrogate that predicts the gas from the feed;
    5. forecast tomorrow's feed;
    6. build the flex-pse model: feed, tank, digester, CHP and grid, under a
       time-of-use tariff;
    7. compare feeding the digester as the sludge arrives with a schedule
       the optimizer chooses.

    It needs flex-pse with the ARIMA surrogate, which is on flex-pse `main`
    today and comes in the next release after 0.3.1:

    ```bash
    pip install "flex-pse[parameterize,solvers] @ git+https://github.com/flex-pse/flex-pse.git@main"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The plant and its data

    `synthetic.digester` simulates the plant, hourly, with the physics a
    digester has:

    - **Feed** (m³/h) follows a daily pattern, highest in the early
      afternoon, at a level that changes from day to day. A few times the
      feed stops for 4 to 12 hours.
    - **Biogas** comes from two pools of digestible feed. A fast pool turns
      feed into gas within a couple of hours, so the gas follows the feed
      closely. A slow pool gives a nearly constant baseline that decays over
      days, so when the feed stops the gas drops, but not to zero.
    - The **online gas meter** (`biogas`) reports every hour but reads
      15 m³/h too high.
    - A **lab** measures the gas accurately three times a day, at 08:00,
      12:00 and 16:00 (`biogas_lab`, empty otherwise).

    The generator knows the true gas rate (`case.underlying`), so every step
    below is scored against it. On top of its faults, we add some of our
    own: spikes and a stuck run on the meter, and a few gaps in the feed.
    The day to schedule is 22 January; the data runs one day past it.
    """)
    return


@app.cell
def _(pd, synthetic):
    # 23 days of hourly data from 1 January 2025 (UTC), with sensor faults.
    case = (
        synthetic.digester(seed=6, periods=23 * 24)
        .with_spikes("biogas", count=6)
        .with_stuck("biogas", length=6)
        .with_gaps("feed", count=3, length=2)
    )
    raw = case.signal()
    DAY = pd.Timestamp("2025-01-22", tz="UTC")  # the day to schedule
    STEP = pd.Timedelta("15min")  # the model's time step
    return DAY, STEP, case, raw


@app.cell
def _(COLORS, case, plt, raw, style):
    _obs = raw.values
    _fig, (_ax, _bx) = plt.subplots(
        2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]}
    )
    _ax.plot(_obs["feed"], color=COLORS["signal"], lw=0.8, label="feed")
    style(_ax, "feed [m³/h]").set_title(
        "The digester's data, 1 to 23 January", loc="left", fontsize=10
    )
    _bx.plot(
        case.underlying["biogas"],
        color=COLORS["truth"],
        lw=2,
        alpha=0.6,
        label="true gas rate",
    )
    _bx.plot(_obs["biogas"], color=COLORS["fault"], lw=0.7, label="online meter")
    _bx.plot(
        _obs["biogas_lab"].dropna(),
        ".",
        ms=4,
        color=COLORS["filled"],
        label="lab",
    )
    style(_bx, "biogas [m³/h]")
    _bx.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(case, mo):
    _feed, _gas = case.underlying["feed"], case.underlying["biogas"]
    _stopped = _feed == 0
    outage_gas_share = float(_gas[_stopped].min() / _gas[~_stopped].mean())
    meter_bias = float((case.observed["biogas"] - case.underlying["biogas"]).median())
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The meter runs parallel to the true gas rate, about
    **{meter_bias:.0f} m³/h** above it, while the lab points sit on it. In the
    feed outages the gas falls, but its lowest value is still
    **{outage_gas_share:.0%}** of the usual rate: that is the slow pool. The
    spikes and the flat stuck run are hard to see at this scale; the
    cleaning finds them.
    """)
    return meter_bias, outage_gas_share


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clean, correct, and put on 15-minute steps

    One pipeline does all of the data work, in this order:

    1. **Hygiene.** `Range` rejects impossible values, `Stuck` a meter that
       repeats one reading for 2 hours or more, and `Hampel` spikes. Its
       6-hour window spans a quarter of the daily cycle, so the cycle itself
       isn't flagged. Rejected values become missing.
    2. **Calibrate** the meter against the lab (the next section).
    3. **Align** everything to 15 minutes, the model's step. `Resample`
       can't do this: it only aggregates, so a 15-minute bin with no hourly
       sample in it would be missing, and an imputer would then label three
       points in four as imputed. `Align(method="linear")` interpolates the
       rates between the hourly samples and flags those points `RESAMPLED`.
       `tolerance="1h"` lets it bridge one hourly spacing and no more, so a
       rejected sample leaves a hole for the imputers rather than being
       interpolated over.
    4. **Impute** what is left: short holes by interpolation, longer ones
       (the stuck run, the feed gaps) by a regression on the daily cycle,
       and for the gas also on the feed. The gas needs the feed already
       filled, so it is a second `Tiered` step.
    """)
    return


@app.cell
def _(fs, raw):
    DAILY = [{"period": "1D", "harmonics": 3}]
    pipe = fs.Pipeline(
        [
            fs.clean.Range("feed", min=0.0, max=30.0),
            fs.clean.Range("biogas", min=0.0, max=600.0),
            fs.clean.Stuck("biogas", min_duration="2h"),
            fs.clean.Hampel("biogas", window="6h", n_sigma=4.0),
            # Correct the meter on its own hourly samples (next section).
            fs.correct.Calibrate(
                target="biogas", reference="biogas_lab", output="biogas_cal"
            ),
            fs.select.Align(freq="15min", method="linear", tolerance="1h"),
            fs.impute.Tiered(
                "feed",
                tiers=[
                    {"max_gap": "1h", "method": "interpolate"},
                    {
                        "max_gap": "12h",
                        "method": "regression",
                        "options": {"fourier": DAILY},
                    },
                ],
            ),
            fs.impute.Tiered(
                "biogas_cal",
                tiers=[
                    {"max_gap": "1h", "method": "interpolate"},
                    {
                        "max_gap": "12h",
                        "method": "regression",
                        "options": {"features": ["feed"], "fourier": DAILY},
                    },
                ],
            ),
        ],
        name="digester_cleanup",
    )
    result = fs.run(pipe.spec, raw)
    clean = result.signal
    return DAILY, clean, pipe, result


@app.cell
def _(clean, pd):
    flag_table = pd.DataFrame(
        [
            {
                "stream": _stream,
                "rows": _c["rows"],
                "valid": _c["valid"],
                "flags": ", ".join(f"{k}={v}" for k, v in sorted(_c["flags"].items())),
            }
            for _stream, _c in clean.flag_counts().items()
        ]
    ).set_index("stream")
    flag_table
    return (flag_table,)


@app.cell
def _(case, clean, fs, pd):
    # Score the cleaner on the hourly samples, where the faults were injected.
    _hours = case.observed.index
    _rows = []
    for _kind, _flag in (("spike", fs.Flag.OUTLIER), ("stuck", fs.Flag.STUCK)):
        _injected = case.fault_mask(_kind, "biogas")
        _caught = clean.has_flag(_flag)["biogas"].reindex(_hours)
        _rows.append(
            {
                "fault in biogas": _kind,
                "injected": int(_injected.sum()),
                "caught": int((_caught & _injected).sum()),
                "false alarms": int((_caught & ~_injected).sum()),
            }
        )
    cleaner_scores = pd.DataFrame(_rows).set_index("fault in biogas")
    cleaner_scores
    return (cleaner_scores,)


@app.cell
def _(case, clean, mo):
    _imputed = clean.flag_counts()
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The {len(case.observed)} hourly rows became **{len(clean)}** 15-minute
    rows. Every injected spike and stuck sample was caught, with no false
    alarms. The {int(case.fault_mask("gap", "feed").sum())} missing hourly
    feed samples left {_imputed["feed"]["flags"]["MISSING"]} 15-minute
    points that `Align` couldn't bridge, and the imputer filled them all
    (`IMPUTED_MODEL`). `biogas_lab` stays sparse: it is a reference, not an
    input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correct the meter against the lab

    `Calibrate` fits a line from the meter's reading to the lab's on the rows
    where both exist, then applies it everywhere and writes the result to a
    new stream, `biogas_cal`. On the lab rows themselves it keeps the lab's
    own value (`prefer_reference`, the default), since that is the better
    measurement. Tutorial 5 covers the options.

    **Where it goes in the pipeline matters.** The lab samples are taken on
    the hour, exactly when the meter takes its hourly sample. So the
    calibration runs *before* `Align`, on the data as measured: every lab
    value is paired with a real meter reading from the same moment, never an
    interpolated one, and the rejected spikes are already out of the fit.
    The corrected stream then goes through `Align` and the imputers like any
    other. (If your lab timestamps didn't match the meter's, you would align
    first, with `method="nearest"`, so they did.)

    Here is how well it worked, against the true gas rate:
    """)
    return


@app.cell
def _(case, clean, np, pd, result):
    _truth = case.underlying["biogas"]
    _faulty = case.fault_mask("spike", "biogas") | case.fault_mask("stuck", "biogas")
    _meter = case.observed["biogas"][~_faulty]
    _corrected = clean.values["biogas_cal"].reindex(_truth.index)
    calibration_fit = next(
        s["fit_metrics"]
        for s in result.report["fit"]["steps"]
        if s["kind"] == "calibrate"
    )
    calibration_scores = pd.DataFrame(
        {
            "RMSE vs truth [m³/h]": [
                float(np.sqrt(((_meter - _truth[_meter.index]) ** 2).mean())),
                float(np.sqrt(((_corrected - _truth) ** 2).mean())),
            ],
            "mean error [m³/h]": [
                float((_meter - _truth[_meter.index]).mean()),
                float((_corrected - _truth).mean()),
            ],
        },
        index=[
            "raw meter (fault-free hours only)",
            "biogas_cal (every hour, filled ones included)",
        ],
    ).round(2)
    calibration_scores
    return calibration_fit, calibration_scores


@app.cell
def _(COLORS, case, clean, fs, pd, plt, raw, style):
    # Three days around the stuck run.
    _start = case.fault_mask("stuck", "biogas").idxmax() - pd.Timedelta("1D")
    _span = slice(_start, _start + pd.Timedelta("3D"))
    _obs = raw.values.loc[_span]
    _out = clean.values.loc[_span]
    _rejected = (
        clean.has_flag(fs.Flag.OUTLIER | fs.Flag.STUCK)["biogas"]
        .reindex(_obs.index)
        .fillna(False)
    )
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.plot(
        case.underlying["biogas"].loc[_span],
        color=COLORS["truth"],
        lw=3,
        alpha=0.6,
        label="true gas rate",
    )
    _ax.plot(_obs["biogas"], color=COLORS["fault"], lw=0.8, label="online meter")
    _ax.plot(
        _obs["biogas"][_rejected],
        "o",
        ms=5,
        mfc="none",
        color=COLORS["fault"],
        label="rejected",
    )
    _ax.plot(_out["biogas_cal"], color=COLORS["signal"], lw=1.1, label="biogas_cal")
    _ax.plot(
        _obs["biogas_lab"].dropna(), ".", ms=6, color=COLORS["filled"], label="lab"
    )
    style(_ax, "biogas [m³/h]").set_title(
        "Cleaned and corrected, at 15 minutes", loc="left", fontsize=10
    )
    _ax.legend(
        frameon=False, ncol=5, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.08)
    )
    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(calibration_fit, calibration_scores, mo):
    _raw, _cal = calibration_scores["RMSE vs truth [m³/h]"]
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    The fit used **{calibration_fit["rows"]}** lab samples
    (R² = {calibration_fit["r2"]:.3f}). The meter's offset is gone: its
    mean error falls from {calibration_scores["mean error [m³/h]"].iloc[0]:.1f}
    to {calibration_scores["mean error [m³/h]"].iloc[1]:.1f} m³/h, and the RMSE
    from **{_raw:.1f}** to **{_cal:.1f} m³/h**, about the meter's own noise.
    That holds even though the corrected stream includes the hours filled in
    for the stuck run.

    ### The same pipeline as JSON

    The pipeline is a spec, so it can be stored as JSON and run from there.
    The JSON gives the same result and the same hash:
    """)
    return


@app.cell
def _(clean, fs, json, mo, pipe, raw):
    spec_text = json.dumps(fs.dump_spec(pipe.spec), indent=2)
    from_json = fs.run(spec_text, raw).signal
    json_parity = from_json.equals(clean) and fs.spec_hash(
        fs.load_spec(spec_text)
    ) == fs.spec_hash(pipe.spec)
    # Display only (marimo): text above a collapsible section.
    mo.vstack(
        [
            mo.md(f"Same result and hash from the JSON: **{json_parity}**"),
            mo.accordion(
                {
                    f"The spec: {len(spec_text.splitlines())} lines of JSON": mo.md(
                        f"```json\n{spec_text}\n```"
                    )
                }
            ),
        ]
    )
    return (json_parity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## An ARIMA surrogate: gas from feed

    The flex-pse `Digestor` needs a relation from its feed to its gas.
    flex-pse's `ArimaRegressor` fits one: a regression of the gas on the
    feed, with ARIMA errors that carry the digester's own slow swings from
    one step to the next.

    `handoff.fit_data` prepares the data:

    - `start` and `end` take the 15 days before the model day (`end` is
      exclusive);
    - `rename` gives the columns the names of the `Digestor`'s own
      variables, because a surrogate's inputs and output must be model
      variables;
    - `timezone="UTC"` returns the index on the plant's wall clock, without
      a zone, as flex-pse's model time is. This plant's clock is UTC;
    - `gaps="error"`: the data is clean already, so anything missing is a
      mistake.

    The first 14 days fit the model and the last one scores it. On the
    held-out day the surrogate runs free from the end of its training data,
    on the measured feed, as it will in the model. Its baseline is
    yesterday's gas profile, repeated.
    """)
    return


@app.cell
def _(ArimaRegressor, DAY, clean, fs, handoff, pd):
    IN_VAR, OUT_VAR = "flow_in_feed", "biogas_volume"  # Digestor variables
    ORDER = (1, 0, 1)
    fit = handoff.fit_data(
        clean.select(["feed"]),
        clean.select(["biogas_cal"]),
        handoff.FitPrep(
            start=(DAY - pd.Timedelta("15D")).isoformat(),
            end=DAY.isoformat(),
            rename={"feed": IN_VAR, "biogas_cal": OUT_VAR},
            timezone="UTC",
            gaps="error",
        ),
    )
    X_train, y_train, X_test, y_test = handoff.split_holdout(fit.X, fit.y, "1D")
    holdout_reg = ArimaRegressor(order=ORDER).fit(
        X_train, y_train, input_units=fit.input_units, output_units=fit.output_units
    )
    # Free-run over the held-out day, on its measured feed.
    arima_pred = pd.Series(
        holdout_reg.model.predict(steps=len(X_test), exog=X_test.to_numpy()),
        index=y_test.index,
    )
    yesterday = pd.Series(
        y_train[OUT_VAR].iloc[-len(y_test) :].to_numpy(), y_test.index
    )
    holdout_scores = pd.DataFrame(
        {
            "ARIMA surrogate": fs.eval.metrics(y_test[OUT_VAR], arima_pred),
            "yesterday's profile": fs.eval.metrics(y_test[OUT_VAR], yesterday),
        }
    ).T.round(3)
    holdout_scores
    return (
        IN_VAR,
        ORDER,
        OUT_VAR,
        X_test,
        arima_pred,
        fit,
        holdout_reg,
        holdout_scores,
        y_test,
        yesterday,
    )


@app.cell
def _(COLORS, OUT_VAR, X_test, arima_pred, plt, style, y_test, yesterday):
    _fig, _ax = plt.subplots(figsize=(9, 3.0))
    _ax.plot(y_test[OUT_VAR], color=COLORS["truth"], lw=2.5, label="biogas_cal")
    _ax.plot(arima_pred, color=COLORS["signal"], lw=1.3, label="ARIMA surrogate")
    _ax.plot(
        yesterday, color=COLORS["fault"], lw=1.0, ls="--", label="yesterday's profile"
    )
    _feed = _ax.twinx()
    _feed.plot(X_test, color=COLORS["filled"], lw=0.8, alpha=0.6)
    _feed.set_ylabel("feed [m³/h] (green)", color=COLORS["filled"], fontsize=9)
    _feed.tick_params(colors=COLORS["filled"], labelsize=8)
    style(_ax, "biogas [m³/h]").set_title(
        f"The held-out day, {y_test.index[0].date()}", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The surrogate follows the day's feed; yesterday's profile can't, because
    the feed level changes from day to day.

    For the model, the regressor is fitted again on all 15 days, so the
    history it stores ends right before the model day. The surrogate starts
    the day from that state: the last 24 hours of residuals and errors,
    dated by `start_date`. If the history ended earlier, or a time zone
    shifted it, the model would start from the wrong state. So check it:
    """)
    return


@app.cell
def _(ArimaRegressor, DAY, ORDER, STEP, fit, mo, pd):
    reg = ArimaRegressor(order=ORDER).fit(
        fit.X, fit.y, input_units=fit.input_units, output_units=fit.output_units
    )
    surrogate = reg.to_surrogate_spec(history_length="1D")
    _history = surrogate.data["history"]
    _kept = len(_history["eta_values"]) - max(ORDER[0] + ORDER[1], ORDER[2])
    history_start = pd.Timestamp(_history["start_date"])
    history_end = history_start + (_kept - 1) * pd.Timedelta(
        seconds=_history["time_step_seconds"]
    )
    model_start = DAY.tz_localize(None)  # the model's clock is naive
    assert history_end + STEP == model_start, "history must end one step before"
    # Display only (marimo): render the result as formatted text.
    mo.md(
        f"History {history_start} to {history_end} ({_kept} steps); the model "
        f"day starts {model_start}. Coefficients: "
        f"`{ {k: round(v, 3) for k, v in reg.coefficients.items()} }`"
    )
    return history_end, model_start, reg, surrogate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tomorrow's feed

    The plan is made the day before, when tomorrow's feed isn't known, so
    the schedule also runs on a forecast. `fs.forecast.Regression` fits the
    daily cycle as sine and cosine terms. It runs at the end of the same
    cleaning pipeline, on the data up to the model day only.

    Two details:

    - The outages are events, not part of the daily pattern, and a few days
      of zeros would drag the fitted cycle down. `Mask` excludes the feed
      where it is zero, so the forecaster learns from the days the plant was
      fed.
    - The last hourly sample before the model day is at 23:00, and `Align`
      can't extend past the last sample, so the forecast is issued at 23:00
      and runs 25 hours to cover the whole day.

    `fs.backtest` scores it on the four days before the model day against
    `SeasonalNaive` (the same time yesterday). Each fold hides everything
    after its issue time, cleaning included.
    """)
    return


@app.cell
def _(DAILY, DAY, fs, pd, pipe, raw):
    HORIZON = "25h"
    forecast_pipe = fs.Pipeline(
        [
            *pipe.spec.steps,
            fs.select.Mask(keep_when="feed > 0", streams=["feed"]),
            fs.forecast.Regression(
                "feed", horizon=HORIZON, fourier=DAILY, quantiles=[0.1, 0.9]
            ),
        ],
        name="feed_forecast",
    )

    def raw_between(start, end):
        """The raw data from start up to (not including) end."""
        window = fs.select.Window(start=start.isoformat(), end=end.isoformat())
        return fs.run(fs.Pipeline([window]).spec, raw).signal

    feed_forecast = fs.run(
        forecast_pipe.spec, raw_between(DAY - pd.Timedelta("14D"), DAY)
    ).signal
    backtest = fs.backtest(
        forecast_pipe.spec,
        raw_between(DAY - pd.Timedelta("21D"), DAY),
        folds=4,
        train="14D",
        compare=[fs.forecast.SeasonalNaive("feed", horizon=HORIZON, season="1D")],
    )
    backtest_summary = backtest.summary().round(3)
    backtest_summary
    return backtest_summary, feed_forecast


@app.cell
def _(COLORS, DAY, backtest_summary, clean, feed_forecast, fs, pd, plt, style):
    _ahead = feed_forecast.has_flag(fs.Flag.FORECAST)["feed"]
    _fc = feed_forecast.values.loc[_ahead]
    _span = slice(DAY - pd.Timedelta("2D"), DAY + pd.Timedelta("1D"))
    forecast_skill = float(
        backtest_summary.set_index("method").loc["forecast_regression", "skill"]
    )
    _fig, _ax = plt.subplots(figsize=(9, 3.0))
    _ax.plot(
        clean.values["feed"].loc[_span],
        color=COLORS["truth"],
        lw=1.6,
        label="cleaned feed (the model day: what will happen)",
    )
    _ax.fill_between(
        _fc.index,
        _fc["feed__q10"],
        _fc["feed__q90"],
        color=COLORS["signal"],
        alpha=0.15,
        lw=0,
        label="80% interval",
    )
    _ax.plot(_fc["feed"], color=COLORS["signal"], lw=1.4, label="forecast")
    _ax.axvline(DAY, color="#a0aec0", lw=0.8)
    style(_ax, "feed [m³/h]").set_title(
        f"Tomorrow's feed; backtest skill vs seasonal-naive {forecast_skill:+.2f}",
        loc="left",
        fontsize=10,
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.autofmt_xdate()
    _fig.tight_layout()
    _fig
    return (forecast_skill,)


@app.cell
def _(feed_forecast, forecast_skill, fs, mo):
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    Issued at **{feed_forecast.meta["feed"].issue_time}**, the forecast fills
    **{int(feed_forecast.has_flag(fs.Flag.FORECAST)["feed"].sum())}**
    15-minute steps. Its RMSE in the backtest is {1 - forecast_skill:.0%} of
    seasonal-naive's (skill {forecast_skill:+.2f}): the day's level is
    random, so copying yesterday copies yesterday's level too, while the
    regression averages two weeks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The flex-pse model

    The flowsheet, for the model day at 15-minute steps:

    **Feed** (the sludge arriving) → **Tank** → **Digestor** (the ARIMA
    surrogate) → **Combustor** (the CHP engine) → **GridConnection** (sells
    the electricity).

    The assumptions:

    - The **tank** holds 8 hours of the average feed, starts half full, and
      stays between 5% and 95% full. It is cyclic: it must end the day at
      least as full as it started, so the plan can't borrow tomorrow's feed.
    - The **digester** may be fed between 50% and 150% of the average feed
      rate, inside the range the surrogate was fitted on.
    - The **biogas** (about 60% methane) holds 6 kWh/m³, and the **CHP**
      turns 35% of that into electricity.
    - The site has no load of its own: it only sells. The **tariff** pays
      0.06 \$/kWh for exports, and 0.25 \$/kWh from 16:00 to 21:00. It is a
      standard EECO tariff file, written to a temporary folder here.
    """)
    return


@app.cell
def _(Path, clean, model_start, pd, surrogate, tempfile):
    TARIFF_ROWS = [  # (type, name, hour start, hour end, $/kWh)
        ("energy", "peak", 16, 21, 0.30),
        ("energy", "offpeak", 0, 16, 0.12),
        ("energy", "offpeak", 21, 24, 0.12),
        ("export", "export-peak", 16, 21, 0.25),
        ("export", "export-offpeak", 0, 16, 0.06),
        ("export", "export-offpeak", 21, 24, 0.06),
    ]
    TARIFF = Path(tempfile.mkdtemp()) / "tariff_biogas_tou.csv"
    TARIFF.write_text(
        "utility,type,name,month_start,month_end,weekday_start,weekday_end,"
        "hour_start,hour_end,basic_charge_limit (metric),charge (metric),units\n"
        + "".join(
            f"electric,{kind},{name},1,12,0,6,{h0},{h1},0,{price},$/kWh\n"
            for kind, name, h0, h1, price in TARIFF_ROWS
        )
    )

    _history = clean.values.loc[: model_start.tz_localize("UTC") - pd.Timedelta("1ns")]
    MEAN_FEED = float(_history["feed"][_history["feed"] > 0].mean())  # m³/h, fed
    TANK_MAX = 8.0 * MEAN_FEED  # m³
    TANK_INIT = 0.5 * TANK_MAX
    LEVEL_MIN, LEVEL_MAX = 0.05, 0.95
    FEED_LO, FEED_HI = 0.5 * MEAN_FEED, 1.5 * MEAN_FEED  # digester feed, m³/h
    BIOGAS_LHV, CHP_EFF = 6.0, 0.35  # kWh/m³, electrical efficiency

    SPEC = {
        "schema_version": "0.1.1",
        "name": "biogas_site",
        "elements": [
            {
                "kind": "time",
                "start_date": model_start.strftime("%Y-%m-%d"),
                "end_date": (model_start + pd.Timedelta("1D")).strftime("%Y-%m-%d"),
                "time_step": "15 min",
            },
            {
                "kind": "property_package",
                "name": "properties",
                "property_class": "SimpleAqueousFlow",
            },
            {
                "kind": "property_package",
                "name": "biogas",
                "property_class": "SimpleGasFlow",
            },
            {
                "kind": "costing",
                "name": "costing",
                "tariff_source": str(TARIFF),
                "decomposition_type": "linear",  # credits exports at export rates
            },
            {"kind": "plant", "name": "site"},
            {
                "kind": "unit",
                "name": "site.feed",
                "unit_model_class": "Feed",
                "property_package": "properties",
                "construction_options": {"outlet_names": ["sludge"]},
                "connections": [{"port": "outlet_sludge", "to": "site.tank.inlet"}],
            },
            {
                "kind": "unit",
                "name": "site.tank",
                "unit_model_class": "Tank",
                "property_package": "properties",
                "construction_options": {
                    "max_volume": {"value": TANK_MAX, "units": "m^3"},
                    "initial_volume": {"value": TANK_INIT, "units": "m^3"},
                    "level_min": LEVEL_MIN,
                    "level_max": LEVEL_MAX,
                    "cyclic": True,
                },
                "connections": [{"port": "outlet", "to": "site.digester.inlet_feed"}],
            },
            {
                "kind": "unit",
                "name": "site.digester",
                "unit_model_class": "Digestor",
                "property_package": None,
                "construction_options": {
                    "inlet_packages": {"feed": {"$package": "properties"}},
                    "biogas_property_package": {"$package": "biogas"},
                    "has_sludge_outlet": False,
                },
                "connections": [
                    {"port": "outlet_biogas", "to": "site.chp.inlet_biogas"}
                ],
            },
            {
                "kind": "unit",
                "name": "site.chp",
                "unit_model_class": "Combustor",
                "property_package": "biogas",
                "construction_options": {
                    "inlet_names": ["biogas"],
                    "heating_values": {
                        "biogas": {"value": BIOGAS_LHV, "units": "kWh/m^3"}
                    },
                    "efficiency": CHP_EFF,
                },
            },
            {
                "kind": "unit",
                "name": "site.grid",
                "unit_model_class": "GridConnection",
                "property_package": None,
                "construction_options": {"max_import": {"value": 0.0, "units": "kW"}},
            },
            # Minimizing the cost maximizes the export revenue.
            {"kind": "objective", "name": "objective"},
        ],
    }
    # The Digestor's biogas_relation becomes the ARIMA surrogate (see below).
    SURROGATE_ELEMENT = {
        "kind": "surrogate",
        "unit": "site.digester",
        "relation": "biogas_relation",
        "surrogate_type": "arima",
        "data": surrogate.data,
    }
    return (
        BIOGAS_LHV,
        CHP_EFF,
        FEED_HI,
        FEED_LO,
        LEVEL_MAX,
        LEVEL_MIN,
        MEAN_FEED,
        SPEC,
        SURROGATE_ELEMENT,
        TANK_INIT,
        TANK_MAX,
        TARIFF,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data goes in the usual way (tutorial 7): `fsp.time_grid` reads the
    model's time steps, `handoff.to_grid` puts the cleaned feed and the
    forecast on them, with `gaps="error"` so nothing is filled silently, and
    `to_overrides(..., strict=True)` names each column by its model path,
    refusing a column that has none. The data is at 15 minutes already, so
    each step takes exactly one sample.

    The tariff gives the price at each step. flex-pse's `price_series` reads
    the import price; the export price is summed from EECO's charge table.
    """)
    return


@app.cell
def _(
    STEP,
    SPEC,
    TARIFF,
    clean,
    eeco_costs,
    feed_forecast,
    fsp,
    handoff,
    load_tariff,
    np,
    price_series,
):
    grid = fsp.time_grid(SPEC)
    GRID = handoff.GridOptions(timezone="UTC", gaps="error")
    measured = handoff.to_grid(clean, grid, GRID, columns=["feed"])
    forecast = handoff.to_grid(feed_forecast, grid, GRID, columns=["feed"])
    FEED_PATH = {"feed": "site.feed.withdrawal"}

    tariff = load_tariff(TARIFF)
    import_price = price_series(tariff, grid).to_numpy()
    _charges = eeco_costs.get_charge_dict(
        grid[0].to_pydatetime(),
        (grid[-1] + STEP).to_pydatetime(),
        tariff,
        resolution="15m",
    )
    export_price = sum(
        np.asarray(_prices)
        for _key, _prices in _charges.items()
        if _key.split("_")[:2] == ["electric", eeco_costs.EXPORT]
    )
    peak = import_price >= import_price.max() - 1e-12
    return FEED_PATH, export_price, forecast, grid, import_price, measured, peak


@app.cell
def _(export_price, forecast, grid, measured, mo, peak):
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    **{len(grid)}** steps from {grid[0]} to {grid[-1]}. The measured feed
    rests on **{measured.resampled["feed"]}** steps that `Align`
    interpolated between hourly samples and **{measured.estimated["feed"]}**
    imputed ones; all **{forecast.estimated["feed"]}** forecast steps are
    estimates. Export prices: {sorted({round(float(p), 2) for p in export_price})}
    \\$/kWh, the high one on {int(peak.sum())} steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things about flex-pse `main` need a workaround here, and each is
    checked rather than assumed:

    - A `surrogate` element of type `arima` in the spec doesn't build yet
      (the spec loader expects the flat coefficients of a multilinear
      surrogate). So the model is built without it, and the surrogate is
      swapped into the `Digestor` with `swap_relation`, which is what the
      element would do.
    - The `Tank` pins its step-0 volume but leaves step 0's inflow and
      outflow out of its balance, so the optimizer can deliver feed at step
      0 for free. The probe below solves the scheduled case without a
      workaround and counts the feed digested beyond what came in. Only if
      there is some does `solve_case` tie step 0's outflow to its inflow.
    """)
    return


@app.cell
def _(
    ArimaSurrogate,
    FEED_HI,
    FEED_LO,
    FEED_PATH,
    SPEC,
    SURROGATE_ELEMENT,
    build_model,
    copy,
    fsp,
    measured,
    pyo,
):
    def build(spec):
        """Build the model, with the ARIMA surrogate as the digester's relation."""
        model = build_model(copy.deepcopy(spec))
        model.site.digester.swap_relation(
            "biogas_relation", ArimaSurrogate(SURROGATE_ELEMENT["data"])
        )
        return model

    def free_delivery(model):
        """Let the tank deliver anything within the digester's feed-rate bounds."""
        # Bounds can't be written in a flex-pse spec, so they are set on the model.
        for t in model.site.tank.flow_out:
            model.site.tank.flow_out[t].setlb(FEED_LO)
            model.site.tank.flow_out[t].setub(FEED_HI)

    try:  # does the spec element build yet?
        build_model(
            copy.deepcopy({**SPEC, "elements": [*SPEC["elements"], SURROGATE_ELEMENT]})
        )
        surrogate_element_builds = True
    except KeyError:
        surrogate_element_builds = False

    _probe = build(
        fsp.apply_values(SPEC, measured.to_overrides(FEED_PATH, strict=True))
    )
    free_delivery(_probe)
    pyo.SolverFactory("appsi_highs").solve(_probe)
    _tank = _probe.site.tank
    free_feed = 0.25 * sum(
        pyo.value(_tank.flow_out[t] - _tank.flow_in[t]) for t in _tank.flow_out
    )  # m³
    TANK_STEP0_BUG = abs(free_feed) > 1e-6
    {
        "surrogate element builds": surrogate_element_builds,
        "feed digested for free [m³]": round(free_feed, 3),
        "tank step-0 workaround needed": TANK_STEP0_BUG,
    }
    return TANK_STEP0_BUG, build, free_delivery, surrogate_element_builds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline against a schedule

    Three cases, each solved with HiGHS:

    - **Baseline:** the digester takes the feed as it arrives (the cleaned
      measurement); the tank only passes it on.
    - **Scheduled, measured inflow:** the delivery from the tank to the
      digester is free within its bounds, and the tank ends where it
      started. This is the best plan if tomorrow's feed were known.
    - **Scheduled, forecast inflow:** the same, planned on the forecast,
      which is what the plant can actually do the day before.
    """)
    return


@app.cell
def _(
    FEED_PATH,
    SPEC,
    TANK_STEP0_BUG,
    build,
    export_price,
    forecast,
    free_delivery,
    fsp,
    import_price,
    measured,
    np,
    pyo,
):
    DT_H = 0.25  # hours per step

    def solve_case(name, inflow, schedule):
        """Fix the tank's inflow, solve, and collect the results."""
        overrides = inflow.to_overrides(FEED_PATH, strict=True)
        if not schedule:  # the tank passes the feed straight on
            overrides["site.tank.flow_out"] = overrides["site.feed.withdrawal"]
        model = build(fsp.apply_values(SPEC, overrides))
        tank = model.site.tank
        if schedule:
            free_delivery(model)
            if TANK_STEP0_BUG:  # flex-pse main leaves step 0 out of the tank balance
                model.t0_balance = pyo.Constraint(
                    expr=tank.flow_out[0] == tank.flow_in[0]
                )
        status = pyo.SolverFactory("appsi_highs").solve(model)
        assert str(status.solver.termination_condition) == "optimal", name

        def series(var):
            return np.array([pyo.value(var[t]) for t in model.time_block.time_index])

        out = {
            "name": name,
            "feed_in": series(model.site.feed.withdrawal),  # m³/h
            "feed_dig": series(tank.flow_out),  # m³/h
            "volume": series(tank.volume),  # m³
            "biogas": series(model.site.digester.biogas_volume),  # m³/h
            "export": series(model.site.grid.grid_export),  # kW
            "import": series(model.site.grid.grid_import),  # kW
            "bill": float(model.costing.report_cost(model).operating.electricity),
        }
        out["revenue_t"] = (
            export_price * out["export"] - import_price * out["import"]
        ) * DT_H
        return out

    cases = [
        solve_case("baseline: fed as it arrives", measured, schedule=False),
        solve_case("scheduled, measured inflow", measured, schedule=True),
        solve_case("scheduled, forecast inflow", forecast, schedule=True),
    ]
    return DT_H, cases


@app.cell
def _(DT_H, cases, pd, peak):
    _base = cases[0]["revenue_t"].sum()
    comparison = pd.DataFrame(
        [
            {
                "case": c["name"],
                "feed digested [m³]": c["feed_dig"].sum() * DT_H,
                "biogas [m³]": c["biogas"].sum() * DT_H,
                "energy sold [kWh]": c["export"].sum() * DT_H,
                "revenue [$]": c["revenue_t"].sum(),
                "peak revenue [$]": c["revenue_t"][peak].sum(),
                "gain [$]": c["revenue_t"].sum() - _base,
                "gain [%]": 100 * (c["revenue_t"].sum() / _base - 1),
            }
            for c in cases
        ]
    ).set_index("case")
    comparison.round(1)
    return (comparison,)


@app.cell
def _(
    CASE_LOOKS,
    COLORS,
    LEVEL_MAX,
    LEVEL_MIN,
    TANK_MAX,
    cases,
    export_price,
    forecast,
    model_start,
    np,
    plt,
    style,
):
    _hours = np.arange(len(export_price)) * 0.25
    _legend = {
        "frameon": False,
        "fontsize": 8,
        "loc": "center left",
        "bbox_to_anchor": (1.01, 0.5),
    }
    _fig, _axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)

    def _peak_band(ax, label=None):
        ax.axvspan(16, 21, color="#edf2f7", zorder=0, label=label)

    _ax = _axes[0]
    _ax.step(
        _hours,
        export_price,
        where="post",
        color=COLORS["filled"],
        lw=1.8,
        label="export price",
    )
    _peak_band(_ax, "peak price (16–21 h)")
    style(_ax, "$/kWh").set_title("What selling a kWh earns", loc="left", fontsize=9)
    _ax.legend(**_legend)

    _ax = _axes[1]
    for _c, _look in zip(cases, CASE_LOOKS, strict=True):
        _ax.step(_hours, _c["feed_dig"], where="post", label=_c["name"], **_look)
    _ax.step(
        _hours,
        np.array(forecast.values["feed"]),
        where="post",
        color=COLORS["truth"],
        lw=0.9,
        ls=":",
        label="forecast inflow",
    )
    _peak_band(_ax)
    style(_ax, "m³/h").set_title(
        "Feed delivered to the digester (baseline = measured inflow)",
        loc="left",
        fontsize=9,
    )
    _ax.legend(**_legend)

    _ax = _axes[2]
    for _c, _look in zip(cases, CASE_LOOKS, strict=True):
        _ax.plot(_hours, _c["volume"], label=_c["name"], **_look)
    for _level, _text in ((LEVEL_MIN, "min"), (LEVEL_MAX, "max")):
        _ax.axhline(
            _level * TANK_MAX,
            color="#c53030",
            lw=0.8,
            ls="-.",
            label=f"tank {_text} ({_level:.0%} full)",
        )
    _peak_band(_ax)
    style(_ax, "m³").set_title("Feed tank level", loc="left", fontsize=9)
    _ax.legend(**_legend)

    _ax = _axes[3]
    for _c, _look in zip(cases, CASE_LOOKS, strict=True):
        _ax.step(_hours, _c["export"], where="post", label=_c["name"], **_look)
    _peak_band(_ax)
    style(_ax, "kW").set_title("Electricity sold by the CHP", loc="left", fontsize=9)
    _ax.set_xlabel(f"hour of {model_start.date()} (plant clock, UTC)")
    _ax.set_xlim(0, 24)
    _ax.set_xticks(range(0, 25, 3))
    _ax.legend(**_legend)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(DT_H, LEVEL_MAX, LEVEL_MIN, TANK_INIT, TANK_MAX, cases, measured, mo, np):
    # The forecast plan, run on the day that actually happened: the delivery is
    # as planned, so the tank absorbs the forecast's error.
    _actual_in = np.array(measured.values["feed"])
    forecast_plan_level = TANK_INIT + np.cumsum(
        (_actual_in - cases[2]["feed_dig"]) * DT_H
    )
    forecast_plan_fits = bool(
        forecast_plan_level.min() >= LEVEL_MIN * TANK_MAX
        and forecast_plan_level.max() <= LEVEL_MAX * TANK_MAX
    )
    _base = cases[0]["revenue_t"].sum()
    _gain = [c["revenue_t"].sum() - _base for c in cases]
    _extra = (cases[2]["feed_in"].sum() - cases[1]["feed_in"].sum()) * DT_H
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    With tomorrow's feed known, the schedule digests exactly the same feed
    and makes exactly the same gas as the baseline; only the timing changes.
    It delivers at the top of the digester's range through the whole
    16:00-21:00 peak, and draws the tank down to pay for it. The revenue
    rises by **{_gain[1]:.0f} \\$** ({_gain[1] / _base:+.1%}). Outside the
    peak the export price is flat, so when the rest of the feed goes in
    doesn't change the revenue; the optimizer picks one of many equally good
    patterns, and only the tank's band and its end level constrain it.

    Planned on the forecast, the gain looks larger, {_gain[2]:.0f} \\$, but
    only because the forecast expects {_extra:.1f} m³ more feed than arrived,
    and the plan digests it. Run on the real day, that plan would take the
    tank down to {forecast_plan_level.min():.1f} m³, below its minimum of
    {LEVEL_MIN * TANK_MAX:.1f} m³. A day-ahead plan is a starting point: in
    practice the plant re-solves it during the day from the measured tank
    level, as the feed arrives. The comparison to trust is the measured
    case.

    **A caveat.** The ARIMA surrogate is a regression with ARIMA errors: the
    gas at a step depends on the feed *in that same step*. A real digester
    responds over hours (the fast pool) and days (the slow pool), so moving
    feed into the peak moves less gas into it than the surrogate says. The
    gain here is an upper bound for this surrogate, not an estimate of the
    real plant's. A surrogate with lagged feed would tell more, and the
    plant's own response sets how far ahead the feed must move.
    """)
    return forecast_plan_fits, forecast_plan_level


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next

    - Tutorial 5 covers `Calibrate` in depth: `fit_when`, robust fits and
      rolling refits for a drifting meter.
    - Tutorial 7 covers the plant's time zone in `to_grid` and `fit_data`;
      this plant ran on UTC.
    - Tutorial 8 covers forecasters, intervals and backtests.
    - In flex-orchestrator, the same calls run as a study: `model_inputs`
      uses `to_grid` and `fit_surrogate` uses `fit_data`, with the same
      options.
    """)
    return


@app.cell
def _(
    BIOGAS_LHV,
    CHP_EFF,
    DT_H,
    FEED_HI,
    FEED_LO,
    IN_VAR,
    LEVEL_MAX,
    LEVEL_MIN,
    MEAN_FEED,
    STEP,
    TANK_INIT,
    TANK_MAX,
    calibration_fit,
    calibration_scores,
    case,
    cases,
    clean,
    cleaner_scores,
    comparison,
    fit,
    flag_table,
    forecast,
    forecast_plan_fits,
    forecast_plan_level,
    forecast_skill,
    feed_forecast,
    fs,
    history_end,
    holdout_scores,
    json_parity,
    measured,
    meter_bias,
    model_start,
    np,
    outage_gas_share,
    peak,
    reg,
    surrogate_element_builds,
):
    # The tutorial's claims, checked.
    # The data: the meter reads ~15 m³/h high; the slow pool keeps gas flowing.
    assert 13 < meter_bias < 17
    assert 0.2 < outage_gas_share < 0.9
    # Cleaning: every spike and stuck sample, no false alarms; every gap filled.
    assert (cleaner_scores["caught"] == cleaner_scores["injected"]).all()
    assert (cleaner_scores["false alarms"] == 0).all()
    assert clean.values[["feed", "biogas_cal"]].notna().all().all()
    assert len(clean) == (len(case.observed) - 1) * 4 + 1
    assert "IMPUTED_MODEL" in flag_table.loc["feed", "flags"]
    # Calibration removed the offset.
    _raw_rmse, _cal_rmse = calibration_scores["RMSE vs truth [m³/h]"]
    assert _raw_rmse > 14 and _cal_rmse < 3 and _cal_rmse < _raw_rmse / 5
    assert abs(calibration_scores["mean error [m³/h]"].iloc[1]) < 1
    assert calibration_fit["r2"] > 0.99
    assert json_parity
    # The ARIMA surrogate beats yesterday's profile, and more feed means more gas.
    assert (
        holdout_scores.loc["ARIMA surrogate", "rmse"]
        < holdout_scores.loc["yesterday's profile", "rmse"]
    )
    assert reg.coefficients[f"exog.{IN_VAR}"] > 0
    assert len(fit.X) == 15 * 96 and fit.X.index.tz is None
    assert fit.X.index[-1] + STEP == model_start == history_end + STEP
    # The forecast: issued at 23:00, 100 steps, beats seasonal-naive.
    assert feed_forecast.meta["feed"].issue_time == "2025-01-21T23:00:00+00:00"
    assert int(feed_forecast.has_flag(fs.Flag.FORECAST)["feed"].sum()) == 100
    assert forecast_skill > 0.2
    assert forecast.estimated["feed"] == 96 and measured.resampled["feed"] > 0
    assert int(peak.sum()) == 20  # 16:00-21:00 at 15 minutes
    # The workaround for the surrogate element is still needed (drop it when not).
    assert not surrogate_element_builds
    # The cases.
    base, sched, sched_fc = cases
    _surrogate_day = reg.model.predict(
        steps=96, exog=np.array(measured.values["feed"]).reshape(-1, 1)
    )
    assert np.allclose(base["biogas"], _surrogate_day, rtol=1e-6, atol=1e-6)
    assert np.allclose(base["feed_dig"], measured.values["feed"])
    assert np.allclose(base["volume"], TANK_INIT)
    for _c in cases:
        assert np.isclose(-_c["bill"], _c["revenue_t"].sum(), rtol=1e-4)
        assert np.allclose(_c["export"], BIOGAS_LHV * CHP_EFF * _c["biogas"])
        assert _c["volume"].min() >= LEVEL_MIN * TANK_MAX - 1e-6
        assert _c["volume"].max() <= LEVEL_MAX * TANK_MAX + 1e-6
    for _c in (sched, sched_fc):
        assert np.isclose(_c["feed_dig"].sum(), _c["feed_in"].sum(), rtol=1e-6)
        assert np.isclose(_c["volume"][-1], TANK_INIT, rtol=1e-6)
        assert _c["feed_dig"].min() >= FEED_LO - 1e-9
        assert _c["feed_dig"].max() <= FEED_HI + 1e-9
        assert _c["revenue_t"].sum() >= base["revenue_t"].sum() - 1e-9
        assert _c["revenue_t"][peak].sum() > base["revenue_t"][peak].sum()
    # Same feed and gas as the baseline, moved into the peak at full rate.
    assert np.isclose(sched["feed_in"].sum(), base["feed_in"].sum())
    assert np.isclose(sched["biogas"].sum(), base["biogas"].sum(), rtol=1e-6)
    assert np.allclose(sched["feed_dig"][peak], FEED_HI)
    assert sched["feed_dig"][peak].mean() > base["feed_dig"][peak].mean()
    assert comparison["gain [$]"].iloc[1] > 0
    assert FEED_LO < MEAN_FEED < FEED_HI
    assert FEED_LO >= fit.X[IN_VAR].min() and FEED_HI <= fit.X[IN_VAR].max()
    # The forecast plan: a larger gain only from more expected feed, and on the
    # real day its tank would go below the minimum.
    assert comparison["gain [$]"].iloc[2] > comparison["gain [$]"].iloc[1]
    assert sched_fc["feed_in"].sum() * DT_H > sched["feed_in"].sum() * DT_H + 1
    assert np.isclose(sched_fc["revenue_t"][peak].sum(), sched["revenue_t"][peak].sum())
    assert not forecast_plan_fits
    assert forecast_plan_level.min() < LEVEL_MIN * TANK_MAX
    return


if __name__ == "__main__":
    app.run()
