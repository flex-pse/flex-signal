import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # marimo (`mo`) is only for the notebook: prose, layout and interactive
    # controls. Lines marked "Display only" or "Notebook control" are not
    # flex-signal; in your own script, print the result or leave them out.
    import warnings

    import marimo as mo
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    import flexsignal as fs
    from flexsignal.core import FLAG_DESCRIPTIONS, QUALITY_FLAGS, Flag, flag_names
    from flexsignal.testing import synthetic

    return (
        FLAG_DESCRIPTIONS,
        Flag,
        QUALITY_FLAGS,
        flag_names,
        fs,
        mdates,
        mo,
        pd,
        plt,
        synthetic,
        warnings,
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
    # 1. A first look at plant data

    flex-signal cleans, corrects, fills and forecasts plant timeseries for the
    flex-pse ecosystem. These tutorials build up from the basics:

    | # | Tutorial | What you learn |
    |---|---|---|
    | **1** | **A first look** (this one) | `SignalFrame`, units, quality flags, and the synthetic test data |
    | 2 | Cleaning one stream | Each cleaning step, how to tune it, and why order matters |
    | 3 | Pipelines and specs | Saving a cleaning recipe as JSON, fitting steps, provenance |
    | 4 | Several streams | Aligning sources, masking operating periods, conditional rules |

    Later releases add tutorials for calibration, model-based imputation,
    flex-pse models and forecasting.

    Every tutorial uses **synthetic data**. Because we inject each fault
    ourselves, we know exactly where it is, so we can check whether a cleaning
    step caught it.

    **Reading the code.** `mo` is [marimo](https://marimo.io), the notebook
    these tutorials run in. It is used only to show results: `mo.md` renders
    text, `mo.vstack` and `mo.hstack` lay outputs out, and `mo.ui` makes the
    sliders and drop-downs. Each such line has a "Display only" or "Notebook
    control" comment. As in Jupyter, a bare expression on a cell's last line,
    such as `df.head(8)`, is displayed. In your own scripts, print those
    results or leave them out; everything else is ordinary flex-signal and
    pandas code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The synthetic plant

    `synthetic.plant()` makes a week of 5-minute data from a simple pumping
    station:

    - **`flow`** (m³/h): about 100, with a daily and a weekly cycle. It is 0
      while the pump is off.
    - **`pressure`** (bar): rises with flow.
    - **`temperature`** (°C): a daily cycle.
    - **`pump_on`** (0/1): whether the pump runs.
    - **`reference_flow`** (m³/h): a precise meter that only reports for two
      hours a day. Tutorial 4 and the calibration tutorial use it.

    A seed makes it repeatable: the same seed always gives the same data.

    To start the way you would with real data, we take the case out as a
    **plain pandas DataFrame**, shaped like a historian export: a `timestamp`
    column, then one column per sensor.
    """)
    return


@app.cell
def _(synthetic):
    case = synthetic.plant(seed=7, pump_off_periods=2)
    df = case.observed.reset_index(names="timestamp")
    df.head(8)
    return case, df


@app.cell
def _(df, mo):
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    Nothing here is specific to flex-signal: `df` is an ordinary
    `{type(df).__name__}` with {len(df)} rows. The timestamps are already in
    UTC (`{df["timestamp"].dtype}`), and each sensor is a float column. NaN
    means "no reading".
    """)
    return


@app.cell
def _(df, mo):
    # Notebook control (marimo): moving it reruns every cell that uses it.
    stream_picker = mo.ui.dropdown(
        options=[c for c in df.columns if c != "timestamp"],
        value="flow",
        label="Stream to plot",
    )
    stream_picker
    return (stream_picker,)


@app.cell
def _(COLORS, df, plt, stream_picker, style):
    _name = stream_picker.value
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.fill_between(
        df["timestamp"],
        0,
        1,
        where=df["pump_on"] == 0,
        transform=_ax.get_xaxis_transform(),
        color=COLORS["excluded"],
        label="pump off",
    )
    _ax.plot(df["timestamp"], df[_name], color=COLORS["signal"], lw=0.8)
    style(_ax, _name)
    _ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.75, 1.0))
    _ax.set_title(f"One week of {_name}", loc="left", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A SignalFrame: values, flags and units

    flex-signal keeps data in a **`SignalFrame`**. It holds three things that
    always travel together:

    - **values**: one float column per stream, on a UTC time index;
    - **flags**: one integer per value, recording what happened to it;
    - **metadata**: units and descriptions per stream.

    No step ever deletes data. A step that rejects a value sets it to NaN and
    records why in the flags, so you can always see what was removed and why.

    ### Building one from a DataFrame

    `SignalFrame.from_frame` turns a plain DataFrame into a SignalFrame. You
    tell it:

    - **`time_column`**: which column holds the timestamps. They become the
      index. Leave it out if the timestamps are already the index.
    - **`units`**: the units of each stream, in the pint syntax flex-pse uses
      (`"m^3/hr"`, `"degC"`). `pump_on` is a 0/1 state, so it has none.
    - **`timezone`**: the time zone of timestamps that have none (see below).

    It checks the data (real timestamps, no duplicates, numeric columns),
    sorts it by time, and flags every NaN as `MISSING`.
    """)
    return


@app.cell
def _(df, fs):
    sf = fs.SignalFrame.from_frame(
        df,
        time_column="timestamp",
        units={
            "flow": "m^3/hr",
            "pressure": "bar",
            "temperature": "degC",
            "reference_flow": "m^3/hr",
        },
    )
    sf
    return (sf,)


@app.cell
def _(mo, sf):
    # Display only (marimo): stack these outputs one above the other.
    mo.vstack(
        [
            mo.md(f"**Units** (`sf.units`): `{sf.units}`"),
            mo.md("**Values** (`sf.values`), one float column per stream:"),
            sf.values.head(4),
            mo.md(
                "**Flags** (`sf.flags`), one integer per value. 0 means nothing "
                "has happened to the value. `reference_flow` shows 1 (`MISSING`): "
                "that meter only reports between 09:00 and 11:00."
            ),
            sf.flags.head(4),
            mo.md(
                "`sf.to_pandas()` gives you a plain DataFrame back, and "
                "`sf.to_pandas(include_flags=True)` adds a `<stream>__flags` "
                "column per stream."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **A shortcut for the synthetic data.** `case.signal()` does exactly what
    the cell above does, with the case's own units. The later tutorials use
    it to keep the code short.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Timestamps without a time zone

    Our synthetic timestamps carry a time zone (UTC), so `from_frame` needed
    no more. Many historian exports give local wall-clock times with no time
    zone. flex-signal then takes them as **this machine's local time zone**,
    which is usually right when the data comes from the plant you are working
    on, and warns you that it did so. Internally every timestamp is still UTC.
    """)
    return


@app.cell
def _(fs, mo, pd, warnings):
    historian_export = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-06 08:00", periods=4, freq="15min"),
            "flow": [101.2, 103.8, float("nan"), 104.1],
        }
    )
    # Record the warning so the notebook can show it. In a script, Python
    # prints it for you, and you would just call from_frame.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assumed = fs.SignalFrame.from_frame(historian_export, time_column="timestamp")

    own = fs.SignalFrame.from_frame(
        historian_export,
        time_column="timestamp",
        timezone="America/Los_Angeles",
        units={"flow": "m^3/hr"},
    )
    # Display only (marimo): stack these outputs one above the other.
    mo.vstack(
        [
            mo.md(
                f"Without `timezone=`, on this machine "
                f"(`fs.local_timezone()` is `{fs.local_timezone()}`):\n\n"
                f"> {type(caught[0].message).__name__}: {caught[0].message}\n\n"
                f"so 08:00 local became `{assumed.index[0]}`."
            ),
            mo.md(
                "Pass `timezone=` when the data comes from somewhere else, or "
                "when the result must be the same on every machine. With "
                '`timezone="America/Los_Angeles"`, 08:00 becomes 16:00 UTC, and '
                "the NaN is flagged `MISSING` (flag value 1):"
            ),
            own.to_pandas(include_flags=True),
        ]
    )
    return caught, own


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quality flags

    Each flag is one bit, so a value can carry several at once. Cleaning steps
    set the **quality** flags when they remove a value. Other steps set flags
    that only record what they did, such as filling a gap or resampling.
    """)
    return


@app.cell
def _(FLAG_DESCRIPTIONS, QUALITY_FLAGS, pd):
    pd.DataFrame(
        [
            {
                "flag": flag.name,
                "bit": int(flag),
                "removes the value": "yes" if flag & QUALITY_FLAGS else "",
                "meaning": meaning,
                "set by": set_by,
            }
            for flag, (meaning, set_by) in FLAG_DESCRIPTIONS.items()
        ]
    )
    return


@app.cell
def _(Flag, flag_names, mo):
    combined = Flag.STUCK | Flag.IMPUTED_SHORT
    # Display only (marimo): render the result as formatted text.
    mo.md(f"""
    Flags combine with `|`. A value that was frozen, removed, and then filled
    by interpolation carries `Flag.STUCK | Flag.IMPUTED_SHORT`, stored as the
    integer **{int(combined)}**. `flag_names({int(combined)})` turns it back into
    `{flag_names(int(combined))}`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Faults with known answers

    The `with_*` methods inject faults and record exactly where. Each returns
    a new case, so they chain:
    """)
    return


@app.cell
def _(case, pd):
    faulty = (
        case.with_out_of_range("flow", count=4)  # -999 error codes
        .with_stuck("flow", length=36)  # a sensor frozen for 3 hours
        .with_spikes("flow", count=6, size=40)  # isolated one-sample spikes
        .with_gaps("flow", count=4, length=2)  # 10-minute dropouts
    )
    fault_kinds = ["out_of_range", "stuck", "spike", "gap"]
    pd.DataFrame(
        {
            "fault": fault_kinds,
            "samples affected": [
                int(faulty.fault_mask(kind, "flow").sum()) for kind in fault_kinds
            ],
        }
    )
    return fault_kinds, faulty


@app.cell
def _(COLORS, fault_kinds, faulty, mdates, pd, plt, style):
    _fig, _axes = plt.subplots(1, 4, figsize=(11, 3.0), sharey=True)
    _observed, _clean = faulty.observed["flow"], faulty.clean["flow"]
    for _ax, _kind in zip(_axes, fault_kinds, strict=True):
        _mask = faulty.fault_mask(_kind, "flow")
        _first = _mask[_mask].index[0]
        _span = slice(_first - pd.Timedelta("2h"), _first + pd.Timedelta("4h"))
        _ax.plot(_clean[_span], color=COLORS["truth"], lw=2.5, label="fault-free")
        _ax.plot(_observed[_span], color=COLORS["signal"], lw=0.9, label="observed")
        if _kind == "gap":
            _ax.fill_between(
                _mask[_span].index,
                0,
                1,
                where=_mask[_span],
                transform=_ax.get_xaxis_transform(),
                color=COLORS["fault"],
                alpha=0.25,
                step="mid",
            )
        else:
            # -999 is far off the scale, so draw it at the bottom edge.
            _hits = _observed[_span][_mask[_span]].clip(lower=45)
            _ax.plot(
                _hits.index, _hits, "o", ms=4, color=COLORS["fault"], label="fault"
            )
        _ax.set_ylim(40, 170)
        _ax.set_title(_kind.replace("_", " "), loc="left", fontsize=10)
        _ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        _ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        style(_ax, "flow [m³/h]" if _kind == "out_of_range" else None)
    _fig.legend(
        *_axes[0].get_legend_handles_labels(),
        frameon=False,
        ncol=3,
        loc="upper right",
        fontsize=9,
    )
    _fig.suptitle(
        "Each fault, zoomed in (UTC; -999 codes drawn at the bottom edge)",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(case, caught, faulty, fs, own, sf):
    # These checks run when CI runs the tutorial as a script.
    assert case.signal().equals(sf)  # the shortcut builds the same frame
    assert [w.category for w in caught] == [fs.NaiveTimestampWarning]
    assert str(own.index[0]) == "2025-01-06 16:00:00+00:00"
    assert sf.flags.drop(columns="reference_flow").to_numpy().max() == 0
    assert sf.flag_counts()["reference_flow"]["flags"] == {
        "MISSING": sf.values["reference_flow"].isna().sum()
    }
    assert own.flag_counts()["flow"]["flags"] == {"MISSING": 1}
    assert int(faulty.fault_mask("gap", "flow").sum()) == 8
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next

    In **tutorial 2** we remove these faults one step at a time, and score
    each step against the injected faults.
    """)
    return


if __name__ == "__main__":
    app.run()
