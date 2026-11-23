import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # This tutorial needs the [pypots] and [foundation] extras:
    #   pip install "flex-signal[pypots,foundation]"
    # needs-module: pypots
    # needs-module: chronos
    #
    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" are not flex-signal; in your own
    # script, print the result or leave them out.
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import flexsignal as fs
    from flexsignal.eval import benchmark_imputers, summarize
    from flexsignal.testing import synthetic

    return benchmark_imputers, fs, mo, np, plt, summarize, synthetic


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
    # 9. Deep imputation and forecasting

    Deep-learning methods fill gaps and forecast from patterns they learn, and
    foundation models forecast series they have never seen. flex-signal
    reaches them through adapters that are steps like any other:

    - `fs.impute.PypotsImpute` trains a PyPOTS network (SAITS by default) on
      the data it fills;
    - `fs.forecast.ChronosForecast` forecasts with Amazon's pretrained
      Chronos-2, with no training at all;
    - `fs.forecast.PypotsForecast` trains a PyPOTS forecasting network.

    This tutorial compares them with the core methods, using the same
    benchmark and backtest as tutorials 6 and 8. It needs the `[pypots]` and
    `[foundation]` extras; the first run downloads the Chronos-2 weights
    (Apache-2.0) at a pinned revision.
    """)
    return


@app.cell
def _(fs, synthetic):
    # Two weeks of a plant, from 6 January 2025 (UTC): flow and pressure, with
    # daily and weekly cycles.
    case = synthetic.plant(seed=11, periods=14 * 288)
    quarter_hourly = fs.run(
        fs.Pipeline([fs.select.Resample(freq="15min")]).spec, case.signal()
    ).signal.select(["flow", "pressure"])
    hourly = fs.run(
        fs.Pipeline([fs.select.Resample(freq="1h")]).spec, case.signal()
    ).signal.select(["flow"])
    return hourly, quarter_hourly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imputation: SAITS against the tiered imputer

    `PypotsImpute` cuts the data into windows (here one day, every two hours),
    trains SAITS to reconstruct values it hides from itself, and fills the
    gaps; `pressure` goes in as a second channel. The trained weights can't be
    stored as JSON, so the network is trained on the data each time it runs
    and the fitted pipeline stores nothing, like the built-in `knn` imputer.

    `benchmark_imputers` hides one-hour and six-hour stretches of flow and
    scores each imputer on them:
    """)
    return


@app.cell
def _(benchmark_imputers, fs, quarter_hourly, summarize):
    DAILY = [{"period": "1D", "harmonics": 3}]
    imputers = {
        "PypotsImpute (SAITS)": fs.impute.PypotsImpute(
            "flow",
            model="SAITS",
            features=["pressure"],
            window="1D",
            stride="2h",
            epochs=40,
            batch_size=8,
            max_gap="1D",
        ),
        "tiered (built in)": fs.impute.Tiered(
            "flow",
            tiers=[
                {"max_gap": "30min", "method": "interpolate"},
                {"max_gap": None, "method": "kalman", "options": {"seasonal": DAILY}},
            ],
        ),
        "linear (built in)": fs.impute.Interpolate("flow", max_gap="1D"),
    }
    imputation = benchmark_imputers(
        quarter_hourly, imputers, streams=["flow"], gaps=["1h", "6h"], count=4
    )
    imputation_rmse = summarize(imputation)
    imputation_rmse.round(2)
    return (imputation_rmse,)


@app.cell
def _(COLORS, imputation_rmse, np, plt, style):
    _colors = [COLORS["fault"], COLORS["signal"], COLORS["truth"]]
    _fig, _ax = plt.subplots(figsize=(9, 2.6))
    _x = np.arange(len(imputation_rmse.columns))
    for _i, (_name, _row) in enumerate(imputation_rmse.iterrows()):
        _ax.bar(_x + (_i - 1) * 0.26, _row, 0.26, color=_colors[_i], label=_name)
    _ax.set_xticks(_x, [f"{g} gaps" for g in imputation_rmse.columns])
    style(_ax, "RMSE [m³/h]").set_title(
        "Hidden flow filled back: lower is better", loc="left", fontsize=10
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(imputation_rmse, mo):
    _saits = imputation_rmse.loc["PypotsImpute (SAITS)", "6h"]
    _tiered = imputation_rmse.loc["tiered (built in)", "6h"]
    _linear = imputation_rmse.loc["linear (built in)", "6h"]
    mo.md(f"""
    On six-hour gaps SAITS (RMSE {_saits:.2f} m³/h) beats a straight line
    ({_linear:.2f}), but not the tiered imputer ({_tiered:.2f}), whose Kalman
    tier models this plant's daily cycle directly. That is typical of a single
    smooth, regular stream: a model of the structure you know wins with little
    data. Deep imputers pay off with many related streams, patterns too
    irregular to write down, and months of data to learn from. The benchmark
    tells you which case you are in.

    ## Forecasting: Chronos-2 against the core methods

    `ChronosForecast` forecasts zero-shot: the pretrained model reads the
    history and returns quantiles directly, so `quantiles=[0.1, 0.9]` come
    from the model itself. The backtest scores it against seasonal-naive, the
    structural model of tutorial 8, and a PyPOTS DLinear network trained on
    the history, on three past days:
    """)
    return


@app.cell
def _(fs, hourly):
    SEASONS = [{"period": "1D", "harmonics": 2}, {"period": "7D", "harmonics": 1}]
    chronos = fs.forecast.ChronosForecast("flow", horizon="1D", quantiles=[0.1, 0.9])
    forecasting = fs.backtest(
        fs.Pipeline([chronos]).spec,
        hourly,
        folds=3,
        compare=[
            fs.forecast.SeasonalNaive("flow", horizon="1D", season="1D"),
            fs.forecast.Structural(
                "flow", horizon="1D", seasonal=SEASONS, quantiles=[0.1, 0.9]
            ),
            fs.forecast.PypotsForecast(
                "flow", horizon="1D", model="DLinear", lookback="3D", epochs=30
            ),
        ],
    )
    forecast_summary = forecasting.summary()
    forecast_summary[["method", "rmse", "coverage", "skill"]].round(3)
    return forecast_summary, forecasting


@app.cell
def _(COLORS, forecasting, plt, style):
    _rows = forecasting.forecasts
    _last = _rows["fold"].max()
    _colors = {
        "ChronosForecast": COLORS["fault"],
        "forecast_structural": COLORS["signal"],
        "PypotsForecast": "#805ad5",
    }
    _fig, (_ax, _bx) = plt.subplots(
        2, 1, figsize=(9, 4.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    _truth = _rows[(_rows["fold"] == _last) & (_rows["method"] == "ChronosForecast")]
    _ax.plot(
        _truth["time"], _truth["actual"], color=COLORS["truth"], lw=2, label="actual"
    )
    for _method, _color in _colors.items():
        _one = _rows[(_rows["fold"] == _last) & (_rows["method"] == _method)]
        _ax.plot(_one["time"], _one["forecast"], color=_color, lw=1.3, label=_method)
        _bx.plot(_one["time"], _one["forecast"] - _one["actual"], color=_color, lw=1.3)
    style(_ax, "flow [m³/h]").set_title(
        "The last backtest day: forecasts against what happened",
        loc="left",
        fontsize=10,
    )
    _ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    # The errors are a few m³/h on a 60 m³/h swing, so they get their own panel.
    _bx.fill_between(
        _truth["time"],
        _truth["q10"] - _truth["actual"],
        _truth["q90"] - _truth["actual"],
        color=COLORS["fault"],
        alpha=0.15,
        lw=0,
        label="Chronos-2 80% interval",
    )
    _bx.axhline(0, color=COLORS["truth"], lw=1)
    style(_bx, "error [m³/h]")
    _bx.legend(frameon=False, fontsize=8, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(forecast_summary, mo):
    _skill = forecast_summary.set_index("method")["skill"]
    mo.md(f"""
    With no training on this plant at all, Chronos-2 has skill
    {_skill["ChronosForecast"]:.2f} against seasonal-naive. The structural
    model does better ({_skill["forecast_structural"]:.2f}), because it is told
    the daily and weekly cycles, and this synthetic flow follows them almost
    exactly. The DLinear network, trained on two weeks of hours, doesn't beat
    seasonal-naive ({_skill["PypotsForecast"]:.2f}): a deep forecaster needs far
    more history than this. Chronos-2 is the method to try when you can't
    write the structure down, or need a reasonable forecast of a new stream
    today.

    ## Licences

    Foundation models are named by Hugging Face id and pinned revision, so a
    spec gives the same forecast everywhere. flex-signal only ships adapters
    to models with open weights: Chronos-2 (Apache-2.0), and
    `fs.forecast.T0Forecast` for The Forecasting Company's t0-alpha
    (Apache-2.0, the `[t0]` extra, which can't be installed alongside
    `[pypots]`). Models whose licences restrict commercial use are yours to
    add, with `fs.register_forecaster` (tutorial 10), once you have checked
    the licence allows your use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    - The adapters are registered methods, like the ones you write yourself
      in tutorial 10. The step table in the reference lists them all, with the
      extra each needs: pyod, ruptures and stumpy cleaners (`[anomaly]`),
      neuralforecast (`[deep]`), darts and sktime forecasters.
    - Any of them can be a `tiered` tier or a `backtest` method, and their
      fitted pipelines save as JSON like any other.
    """)
    return


@app.cell
def _(forecast_summary, imputation_rmse, np):
    # The tutorial's claims, checked.
    assert np.isfinite(imputation_rmse.to_numpy()).all()
    assert (
        imputation_rmse.loc["PypotsImpute (SAITS)", "6h"]
        < imputation_rmse.loc["linear (built in)", "6h"]
    )
    _skill = forecast_summary.set_index("method")["skill"]
    assert _skill["ChronosForecast"] > 0.5
    assert _skill["forecast_structural"] > _skill["ChronosForecast"]
    return


if __name__ == "__main__":
    app.run()
