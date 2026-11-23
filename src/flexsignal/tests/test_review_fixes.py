"""Regression tests for review findings in the pipeline runner."""

import pytest

import flexsignal as fs
from flexsignal.testing import synthetic


@pytest.mark.unit
def test_a_pipeline_with_no_steps_still_reports_progress_once():
    sf = synthetic.plant(periods=20).signal()
    pipe = fs.Pipeline([], name="empty")
    fit_calls, transform_calls = [], []
    fitted = pipe.fit(sf, progress=lambda f, m: fit_calls.append((f, m)))
    fitted.transform(sf, progress=lambda f, m: transform_calls.append((f, m)))
    assert fit_calls == [(1.0, "fit done")]
    assert transform_calls == [(1.0, "transform done")]


@pytest.mark.unit
def test_progress_with_steps_is_unchanged():
    sf = synthetic.plant(periods=20).signal()
    pipe = fs.Pipeline([fs.clean.Range("flow", min=0)], name="one")
    calls = []
    fitted = pipe.fit(sf, progress=lambda f, m: calls.append((f, m)))
    fitted.transform(sf, progress=lambda f, m: calls.append((f, m)))
    assert calls == [(1.0, "fit range"), (1.0, "transform range")]


@pytest.mark.component
def test_a_forecaster_with_exog_fits_on_history_then_forecasts_new_data():
    import numpy as np

    import flexsignal as fs
    from flexsignal.testing import synthetic

    full = synthetic.plant(seed=4, periods=5 * 288).signal()
    cut = full.index[4 * 288 - 1]
    history = full.values.loc[:cut]  # targets and exog end together
    train = (history, dict(full.units))
    spec = fs.Pipeline(
        [
            fs.clean.Range("flow", min=0),
            fs.forecast.Regression(
                "flow", horizon="6h", fourier=[{"period": "1D"}], exog=["temperature"]
            ),
        ],
        name="ahead",
    ).spec
    fitted = fs.fit(spec, train)
    assert fitted.fit_report["steps"][-1]["applied"] is False
    assert fs.Pipeline.from_spec(spec).fit(train).states == fitted.states
    # Apply: tomorrow's temperature is known, flow isn't.
    new = full.values.loc[: cut + (full.index[1] - full.index[0]) * 72].copy()
    new.loc[new.index > cut, "flow"] = np.nan
    out = fitted.transform((new, dict(full.units)))
    ahead = out.has_flag(fs.Flag.FORECAST)["flow"]
    assert ahead.sum() == 72 and not out.values["flow"][ahead].isna().any()
