import numpy as np
import pytest
from pydantic import ValidationError

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError
from flexsignal.testing import synthetic

TRUSTED = "C == 0 and D == 0 and B > 0"


def _rmse(out, case, stream="A_clean"):
    error = out.values[stream] - case.underlying["A"]
    return float(np.sqrt(np.nanmean(error.to_numpy() ** 2)))


def _fit_transform(step, sf):
    fitted, out = fs.Pipeline([step]).fit_transform(sf)
    return fitted, out


@pytest.mark.component
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrate_recovers_the_true_a(seed):
    # The key calibration check: A_clean matches the true A to within its noise.
    case = synthetic.abcd(seed=seed)
    step = fs.correct.Calibrate(target="A", reference="B", fit_when=TRUSTED)
    fitted, out = _fit_transform(step, case.signal())
    assert _rmse(out, case) < 1.5  # raw A is off by ~10
    assert out.values["A_clean"].notna().all()
    metrics = fitted.states[0]["metrics"]
    assert metrics["r2"] > 0.99
    assert fitted.report()["fit"]["steps"][0]["fit_metrics"] == metrics
    # Fitting on every reference sample, bad ones included, is far worse.
    _, naive = _fit_transform(
        fs.correct.Calibrate(target="A", reference="B"), case.signal()
    )
    assert _rmse(naive, case) > 5


@pytest.mark.component
@pytest.mark.parametrize("seed", [0, 1])
def test_rolling_refit_tracks_drift_that_static_misses(seed):
    case = synthetic.abcd(seed=seed, drift_per_day=2.0)
    static = fs.correct.Calibrate(target="A", reference="B", fit_when=TRUSTED)
    rolling = static.model_copy(update={"refit": fs.correct.RollingRefit(rolling="1D")})
    assert _rmse(_fit_transform(static, case.signal())[1], case) > 5
    assert _rmse(_fit_transform(rolling, case.signal())[1], case) < 2


@pytest.mark.unit
def test_flags_reference_rows_and_apply_when():
    case = synthetic.abcd(seed=0, periods=2 * 288)
    sf = case.signal()
    step = fs.correct.Calibrate(
        target="A", reference="B", fit_when=TRUSTED, apply_when="C == 0"
    )
    _, out = _fit_transform(step, sf)
    trusted = fs.core.evaluate_condition(sf, TRUSTED).to_numpy()
    flags = out.flags["A_clean"].to_numpy()
    values = out.values["A_clean"].to_numpy()
    # Trusted rows take B itself, with B's (raw) flags.
    assert np.array_equal(values[trusted], sf.values["B"].to_numpy()[trusted])
    assert np.all(flags[trusted] == 0)
    # Where apply_when fails, A passes through unchanged and unflagged.
    off = (sf.values["C"] == 1).to_numpy() & ~trusted
    assert np.array_equal(values[off], sf.values["A"].to_numpy()[off])
    rest = ~trusted & ~off
    assert np.all(flags[rest] == int(Flag.CORRECTED))
    assert out.units["A_clean"] == "m^3/hr"
    # Without prefer_reference every applied row is a prediction.
    no_ref = step.model_copy(update={"prefer_reference": False})
    _, out2 = _fit_transform(no_ref, sf)
    assert np.all(out2.flags["A_clean"].to_numpy()[~off] == int(Flag.CORRECTED))


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"model": "sklearn:HuberRegressor"},
        {"model": "sklearn:Ridge", "params": {"alpha": 0.1}},
        {"model": "statsmodels:RLM"},
        {
            "model": "scipy:curve_fit",
            "formula": "a + b * A",
            "initial": {"a": 0, "b": 1},
        },
        {"mode": "residual"},
    ],
)
def test_every_model_calibrates(options):
    case = synthetic.abcd(seed=0, periods=3 * 288)
    step = fs.correct.Calibrate(
        target="A", reference="B", fit_when=TRUSTED, prefer_reference=False, **options
    )
    _, out = _fit_transform(step, case.signal())
    assert _rmse(out, case) < 1.6


@pytest.mark.unit
def test_output_can_replace_the_target_and_json_gives_the_same_result():
    case = synthetic.abcd(seed=0, periods=2 * 288)
    step = fs.correct.Calibrate(target="A", reference="B", output="A", fit_when=TRUSTED)
    spec = fs.Pipeline([step]).spec
    _, out = fs.Pipeline([step]).fit_transform(case.signal())
    _, again = fs.Pipeline.from_spec(fs.dump_spec(spec)).fit_transform(case.signal())
    assert out.columns == ["A", "B", "C", "D"]
    assert out.equals(again)


@pytest.mark.unit
def test_calibrate_rejects_bad_options():
    with pytest.raises(ValidationError, match="unknown model"):
        fs.correct.Calibrate(target="A", reference="B", model="os:system")
    with pytest.raises(ValidationError, match="needs a formula"):
        fs.correct.Calibrate(target="A", reference="B", model="scipy:curve_fit")
    with pytest.raises(ValidationError, match="only used with"):
        fs.correct.Calibrate(target="A", reference="B", formula="a * A")
    with pytest.raises(ValidationError, match="must be different"):
        fs.correct.Calibrate(target="A", reference="A")
    with pytest.raises(SignalConfigError) as info:
        fs.load_spec(
            {
                "schema_version": fs.SCHEMA_VERSION,
                "name": "x",
                "steps": [
                    {
                        "kind": "calibrate",
                        "target": "A",
                        "reference": "B",
                        "model": "subprocess:run",
                    }
                ],
            }
        )
    assert info.value.field == "steps.0.model"
    case = synthetic.abcd(seed=0, periods=288)
    with pytest.raises(fs.SignalDataError, match="no trusted rows"):
        fs.correct.Calibrate(target="A", reference="B", fit_when="B < -1").fit(
            case.signal()
        )
    with pytest.raises(SignalConfigError, match="already exists"):
        fs.correct.Calibrate(target="A", reference="B", output="C").fit(case.signal())
