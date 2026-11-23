"""Regression tests for review findings in the imputer benchmark."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError, SignalDataError
from flexsignal.eval import benchmark_imputers, hide_values
from flexsignal.testing import synthetic


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_random_pattern_never_hides_two_neighbouring_samples(seed):
    sf = synthetic.plant(seed=seed, pump_off_periods=3).signal()
    masked, hidden = hide_values(
        sf, ["flow"], gap="1h", count=20, pattern="random", seed=seed
    )
    where = hidden["flow"].to_numpy()
    assert where.sum() == 20 * 11
    assert not (where[1:] & where[:-1]).any()
    # Every hidden sample sits between two valid ones.
    valid = masked.values["flow"].notna().to_numpy()
    picks = np.flatnonzero(where)
    assert valid[picks - 1].all() and valid[picks + 1].all()


@pytest.mark.unit
def test_random_pattern_is_seeded():
    sf = synthetic.plant(seed=0).signal()
    _, a = hide_values(sf, ["flow"], gap="15min", count=30, pattern="random", seed=4)
    _, b = hide_values(sf, ["flow"], gap="15min", count=30, pattern="random", seed=4)
    _, c = hide_values(sf, ["flow"], gap="15min", count=30, pattern="random", seed=5)
    assert a.equals(b)
    assert not a.equals(c)


@pytest.mark.unit
def test_random_pattern_says_when_there_is_no_room():
    # 10 samples have 8 interior ones, but only 4 can be hidden without touching.
    sf = synthetic.plant(seed=0, periods=10).signal()
    _, hidden = hide_values(sf, ["flow"], gap="10min", count=4, pattern="random")
    assert hidden["flow"].sum() == 4
    with pytest.raises(SignalDataError, match="room for only 4 of 5"):
        hide_values(sf, ["flow"], gap="10min", count=5, pattern="random")


@pytest.mark.unit
@pytest.mark.parametrize("count", [0, -1])
@pytest.mark.parametrize("pattern", ["block", "random"])
def test_count_must_be_at_least_one(count, pattern):
    sf = synthetic.plant(seed=0, periods=100).signal()
    with pytest.raises(SignalConfigError, match="at least 1") as info:
        hide_values(sf, ["flow"], gap="15min", count=count, pattern=pattern)
    assert info.value.field == "count"
    with pytest.raises(SignalConfigError, match="at least 1"):
        benchmark_imputers(
            sf,
            {"linear": fs.impute.Interpolate("flow", max_gap="1D")},
            streams=["flow"],
            gaps=["15min"],
            count=count,
            pattern=pattern,
        )


@pytest.mark.unit
def test_hide_values_needs_a_regular_index():
    sf = synthetic.plant(seed=0, periods=50).signal()
    irregular = sf.values.drop(sf.index[[3, 7]])
    with pytest.raises(SignalDataError, match="regular time index"):
        hide_values(fs.SignalFrame(irregular), ["flow"], gap="15min", count=1)


@pytest.mark.component
def test_benchmark_scores_match_a_hand_computation():
    sf = synthetic.plant(seed=5, periods=3 * 288, pump_off_periods=2).signal()
    imputers = {
        "linear": fs.impute.Interpolate("flow", max_gap="1D"),
        # Leaves the 6 h gaps unfilled: those rows score over nothing.
        "short": fs.impute.Interpolate("flow", max_gap="30min"),
    }
    results = benchmark_imputers(
        sf, imputers, streams=["flow"], gaps=["10min", "6h"], count=3
    )
    for row in results.itertuples():
        g_i = ["10min", "6h"].index(row.gap)
        masked, hidden = hide_values(
            sf, ["flow"], gap=row.gap, count=3, seed=1000 * g_i
        )
        filled = fs.Pipeline([imputers[row.imputer]], name="x").fit_transform(masked)
        where = hidden["flow"].to_numpy()
        truth = sf.values["flow"].to_numpy()[where]
        guess = filled[1].values["flow"].to_numpy()[where]
        got = ~np.isnan(guess)
        error = guess[got] - truth[got]
        if got.any():
            assert row.rmse == pytest.approx(np.sqrt(np.mean(error**2)), rel=1e-12)
            assert row.mae == pytest.approx(np.mean(np.abs(error)), rel=1e-12)
            assert row.bias == pytest.approx(np.mean(error), rel=1e-12)
        else:
            assert np.isnan(row.rmse) and np.isnan(row.mae) and np.isnan(row.bias)
    short_6h = results[(results.imputer == "short") & (results.gap == "6h")]
    assert (short_6h.coverage == 0).all() and short_6h.rmse.isna().all()
    assert isinstance(results, pd.DataFrame)
