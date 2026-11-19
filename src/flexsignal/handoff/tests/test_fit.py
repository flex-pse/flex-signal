"""fit_data: input and output data prepared for a regressor.

flex-orchestrator's fit-preparation tests, ported with the same data shapes
(FLEX_ORCHESTRATOR_OVERLAP.md §4). Their frames have naive timestamps, which
fit_data keeps as they are.
"""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError, SignalDataError
from flexsignal.handoff import FitPrep, data_hash, fit_data, split_holdout


def ro_frame(hours=48, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2025-03-01", periods=hours, freq="h")
    permeate = rng.uniform(30, 150, hours)
    power = 3.0 * permeate + 10 + rng.normal(0, 1, hours)
    return pd.DataFrame({"permeate": permeate, "power_electrical": power}, time)


@pytest.mark.unit
def test_prepare_aligns_and_hashes():
    frame = ro_frame(6)
    xs = [frame[["permeate"]].iloc[1:]]
    ys = [frame[["power_electrical"]]]
    inner = fit_data(xs, ys)
    assert len(inner.X) == 5
    assert inner.steps[0] == {"step": "align", "how": "inner", "rows": 5}
    assert fit_data(xs, ys).data_hash == inner.data_hash == data_hash(inner.X, inner.y)
    changed = fit_data([frame[["permeate"]].iloc[1:] + 1], ys)
    assert changed.data_hash != inner.data_hash
    with pytest.raises(SignalDataError, match="have missing values"):
        fit_data(xs, ys, FitPrep(align="outer"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("gaps", "rows"), [("drop", 5), ("interpolate", 6), ("longest_segment", 3)]
)
def test_gap_handling(gaps, rows):
    frame = ro_frame(6)
    frame.iloc[1 if gaps == "longest_segment" else 2, 0] = np.nan
    if gaps == "longest_segment":
        frame.iloc[5, 0] = np.nan  # complete runs: row 0, rows 2-4
    result = fit_data(
        frame[["permeate"]], frame[["power_electrical"]], {"gaps": gaps, "limit": 1}
    )
    assert len(result.X) == rows
    assert result.steps[-1]["how"] == gaps
    assert not result.X.isna().any().any()


@pytest.mark.unit
def test_interpolation_leaves_ends_and_long_gaps():
    frame = ro_frame(6)
    frame.iloc[0, 0] = np.nan
    with pytest.raises(SignalDataError, match="still have missing values"):
        fit_data(
            frame[["permeate"]], frame[["power_electrical"]], {"gaps": "interpolate"}
        )
    frame.iloc[:, 0] = np.nan
    with pytest.raises(SignalDataError, match="no segment"):
        fit_data(
            frame[["permeate"]],
            frame[["power_electrical"]],
            {"gaps": "longest_segment"},
        )


@pytest.mark.unit
def test_resample_and_guards():
    frame = ro_frame(6)
    xs, ys = [frame[["permeate"]]], [frame[["power_electrical"]]]
    result = fit_data(xs, ys, {"resample": {"freq": "2h", "agg": "max"}})
    assert len(result.X) == 3
    assert result.steps[1]["step"] == "resample"
    with pytest.raises(SignalDataError, match="more than max_rows=2"):
        fit_data(xs, ys, {"max_rows": 2})
    with pytest.raises(SignalConfigError, match="more than one selector"):
        fit_data(xs + xs, ys)
    with pytest.raises(SignalConfigError, match="one column"):
        fit_data(xs, ys + [frame[["permeate"]].rename(columns=str.upper)])
    with pytest.raises(SignalDataError, match="No rows are left"):
        empty = frame.iloc[:0]
        fit_data([empty[["permeate"]]], [empty[["power_electrical"]]])


@pytest.mark.unit
def test_holdout_split():
    frame = ro_frame(48)
    X_, y_ = frame[["permeate"]], frame[["power_electrical"]]
    train, _, test, _ = split_holdout(X_, y_, "12h")
    assert (len(train), len(test)) == (36, 12)
    with pytest.raises(SignalConfigError, match="Can't read holdout"):
        split_holdout(X_, y_, "fortnight")
    with pytest.raises(SignalDataError, match="no rows to train"):
        split_holdout(X_, y_, "60D")


@pytest.mark.unit
def test_metrics():
    actual = pd.Series([0.0, 2.0, 4.0])
    found = fs.eval.metrics(actual, pd.Series([1.0, 2.0, 2.0]))
    assert found["mae"] == pytest.approx(1.0)
    assert found["mape"] == pytest.approx(0.25)
    assert "mape" not in fs.eval.metrics(pd.Series([0.0]), pd.Series([1.0]))


@pytest.mark.unit
def test_signal_frames_bring_their_units():
    frame = ro_frame(6).tz_localize("UTC")
    sf = fs.SignalFrame.from_frame(
        frame, units={"permeate": "m^3/hr", "power_electrical": "kW"}
    )
    found = fit_data(sf.select(["permeate"]), sf.select(["power_electrical"]))
    assert found.units == {"permeate": "m^3/hr", "power_electrical": "kW"}
    as_tuple = fit_data(
        (frame[["permeate"]], {"permeate": "m^3/hr"}), frame[["power_electrical"]]
    )
    assert as_tuple.units == {"permeate": "m^3/hr", "power_electrical": None}
    assert found.data_hash == as_tuple.data_hash
