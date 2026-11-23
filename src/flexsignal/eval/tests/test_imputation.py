import numpy as np
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError, SignalDataError
from flexsignal.eval import benchmark_imputers, hide_values, summarize
from flexsignal.testing import synthetic


@pytest.mark.unit
@pytest.mark.parametrize("pattern", ["block", "random"])
def test_hide_values_hides_exactly_what_it_reports(pattern):
    sf = synthetic.plant(seed=0).signal()
    masked, hidden = hide_values(
        sf, ["flow", "pressure"], gap="1h", count=4, pattern=pattern, seed=3
    )
    for col in ["flow", "pressure"]:
        where = hidden[col].to_numpy()
        assert where.sum() == 4 * 11  # a 1 h gap on 5 min data hides 11 samples
        assert masked.values[col][where].isna().all()
        assert masked.values[col][~where].equals(sf.values[col][~where])
    if pattern == "block":
        runs = np.flatnonzero(np.diff(np.concatenate([[0], where, [0]])) == 1)
        assert len(runs) == 4
        # Each gap is 1 h long as an imputer measures it.
        step = fs.impute.Interpolate("flow", max_gap="1h")
        out = step.transform(masked, {})
        assert out.values["flow"][hidden["flow"]].notna().all()
        short = fs.impute.Interpolate("flow", max_gap="55min").transform(masked, {})
        assert short.values["flow"][hidden["flow"]].isna().all()


@pytest.mark.unit
def test_hide_values_checks_its_inputs():
    sf = synthetic.plant(seed=0, periods=100).signal()
    with pytest.raises(SignalConfigError, match="whole number"):
        hide_values(sf, ["flow"], gap="7min", count=1)
    with pytest.raises(SignalDataError, match="room for only"):
        hide_values(sf, ["flow"], gap="6h", count=3)


@pytest.mark.component
def test_benchmark_ranks_imputers_by_gap_length():
    sf = synthetic.plant(seed=2).signal()
    imputers = {
        "linear": fs.impute.Interpolate("flow", max_gap="2D"),
        "kalman": fs.impute.Kalman("flow", seasonal=[{"period": "1D", "harmonics": 2}]),
        "short_only": fs.impute.Interpolate("flow", max_gap="15min"),
    }
    results = benchmark_imputers(
        sf, imputers, streams=["flow"], gaps=["15min", "6h"], count=5
    )
    assert list(results.columns) == [
        "imputer", "stream", "gap", "repeat", "hidden", "filled", "coverage",
        "rmse", "mae", "bias",
    ]  # fmt: skip
    rmse = summarize(results)
    assert list(rmse.columns) == ["15min", "6h"]
    assert list(rmse.index) == ["linear", "kalman", "short_only"]
    # Interpolation is fine for short gaps but falls behind on long ones.
    assert rmse.loc["linear", "6h"] > 2 * rmse.loc["kalman", "6h"]
    coverage = summarize(results, "coverage")
    assert coverage.loc["short_only", "15min"] == 1.0
    assert coverage.loc["short_only", "6h"] == 0.0
    assert np.isnan(rmse.loc["short_only", "6h"])
