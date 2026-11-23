"""Regression tests for the fixes from the 0.1.0 review of the built-in steps."""

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalConfigError, SignalDataError


def _frame(columns, periods=None, freq="1h", units=None):
    periods = periods or len(next(iter(columns.values())))
    index = pd.date_range("2025-01-01", periods=periods, freq=freq, tz="UTC")
    return fs.SignalFrame.from_frame(pd.DataFrame(columns, index=index), units=units)


def _run(step, sf):
    return fs.run(fs.Pipeline([step], name="review").spec, sf).signal


def _load_and_price(days=10, seed=3):
    """Hourly load driven by a price schedule; the price is known a day ahead."""
    rng = np.random.default_rng(seed)
    n = 24 * days
    price = 0.1 + 0.05 * rng.random(n + 24)
    load = 50 - 200 * price + rng.normal(0, 0.1, n + 24)
    history = _frame({"load": load[:n], "price": price[:n]})
    ahead = load.copy()
    ahead[n:] = np.nan
    new = _frame({"load": ahead, "price": price})
    return history, new, load[n:]


# -- forecast: fitting with exog on plain history -------------------------------------


@pytest.mark.unit
def test_a_regression_forecaster_with_exog_fits_on_history_that_ends_with_the_target():
    history, new, truth = _load_and_price()
    step = fs.forecast.Regression("load", horizon="1D", exog=["price"])
    state = step.fit(history)  # used to need price values a day past the history
    out = step.transform(new, state)
    rows = out.has_flag(Flag.FORECAST)["load"].to_numpy()
    assert rows.sum() == 24
    assert np.allclose(out.values["load"].to_numpy()[rows], truth, atol=1.0)


@pytest.mark.component
def test_a_structural_forecaster_with_exog_fits_on_history_that_ends_with_the_target():
    history, new, _ = _load_and_price()
    step = fs.forecast.Structural("load", horizon="1D", exog="price")
    state = step.fit(history)
    out = step.transform(new, state)
    assert out.has_flag(Flag.FORECAST)["load"].sum() == 24


@pytest.mark.unit
def test_forecasting_still_needs_exog_values_over_the_horizon():
    history, _, _ = _load_and_price()
    step = fs.forecast.Regression("load", horizon="1D", exog=["price"])
    state = step.fit(history)
    with pytest.raises(SignalDataError, match="exog stream 'price' has no value"):
        step.transform(history, state)


@pytest.mark.unit
def test_fitting_with_exog_gives_the_same_model_with_or_without_exog_ahead():
    history, new, _ = _load_and_price()
    step = fs.forecast.Regression(
        "load", horizon="1D", exog=["price"], quantiles=[0.1, 0.9]
    )
    assert step.fit(history) == step.fit(new)


# -- forecast: quantile streams must not overwrite input streams --------------------


@pytest.mark.unit
def test_a_quantile_stream_doesnt_silently_overwrite_an_input_stream():
    rng = np.random.default_rng(0)
    sf = _frame({"y": rng.normal(0, 1, 48), "y__q90": np.full(48, 123.0)})
    step = fs.forecast.Persistence("y", horizon="2h", quantiles=[0.9])
    with pytest.raises(SignalConfigError, match="'y__q90'") as info:
        _run(step, sf)
    assert info.value.field == "quantiles"


@pytest.mark.unit
def test_a_fitted_forecaster_runs_again_on_its_own_output():
    rng = np.random.default_rng(1)
    sf = _frame({"y": np.cumsum(rng.normal(0, 1, 48))})
    issue = sf.index[-1].isoformat()
    step = fs.forecast.Persistence(
        "y", horizon="2h", quantiles=[0.1, 0.9], issue_time=issue
    )
    fitted, first = fs.Pipeline([step], name="q").fit_transform(sf)
    again = fitted.transform(first)  # y__q10 / y__q90 hold only forecast values
    assert again.values.equals(first.values)
    assert again.flags.equals(first.flags)


# -- forecast.Regression quantiles: {q: values} from the residuals ------------------


@pytest.mark.unit
def test_regression_quantiles_are_the_mean_plus_the_residual_quantiles():
    history, new, _ = _load_and_price()
    step = fs.forecast.Regression(
        "load", horizon="1D", exog=["price"], quantiles=[0.1, 0.5, 0.9]
    )
    state = step.fit(history)
    out = step.transform(new, state)
    rows = out.has_flag(Flag.FORECAST)["load"].to_numpy()
    mean = out.values["load"].to_numpy()[rows]
    offsets = state["streams"]["load"]["residual_quantiles"]
    for q, name in [(0.1, "load__q10"), (0.5, "load__q50"), (0.9, "load__q90")]:
        assert np.allclose(out.values[name].to_numpy()[rows], mean + offsets[str(q)])
    assert offsets["0.1"] < 0 < offsets["0.9"]


# -- calibrate: map mode doesn't need the target unless it is an input -------------


def _offline_target(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(10, 1, 200)
    a = x.copy()
    a[::2] = np.nan  # A is offline on every other row...
    b = 2 * x + 1
    b[1::2] = np.nan  # ...and B is trusted only on those rows
    return _frame({"A": a, "B": b, "x": x})


@pytest.mark.unit
def test_calibrate_in_map_mode_fits_where_the_target_is_missing_but_not_an_input():
    sf = _offline_target()
    step = fs.correct.Calibrate(target="A", reference="B", features=["x"])
    fitted, out = fs.Pipeline([step], name="c").fit_transform(sf)
    assert fitted.states[0]["metrics"]["rows"] == 100
    assert np.allclose(
        out.values["A_clean"], 2 * sf.values["x"] + 1, equal_nan=False, atol=1e-6
    )


@pytest.mark.unit
def test_calibrate_still_needs_the_target_when_it_is_an_input():
    sf = _offline_target()
    for step in [
        fs.correct.Calibrate(target="A", reference="B"),  # features default: A
        fs.correct.Calibrate(target="A", reference="B", features=["x", "A"]),
        fs.correct.Calibrate(
            target="A", reference="B", features=["x"], mode="residual"
        ),
    ]:
        with pytest.raises(SignalDataError, match="only 0 rows"):
            fs.Pipeline([step], name="c").fit(sf)


# -- rate_of_change: the documented first-value behaviour -------------------------


@pytest.mark.unit
def test_rate_of_change_anchors_on_the_first_value_even_when_it_is_bad():
    # Documented (docs/reference/steps.rst): run range or hampel first.
    rng = np.random.default_rng(3)
    values = 10.0 + rng.normal(0, 0.1, 48)
    values[0] = 1000.0
    step = fs.clean.RateOfChange("y", max_rate=5, per="1h")
    out = _run(step, _frame({"y": values}))
    assert int(out.has_flag(Flag.RATE_OF_CHANGE)["y"].sum()) == 47
    # With range first, the bad first value can't become the reference.
    pipe = fs.Pipeline([fs.clean.Range("y", max=100), step], name="r")
    out = fs.run(pipe.spec, _frame({"y": values})).signal
    assert int(out.has_flag(Flag.RATE_OF_CHANGE)["y"].sum()) == 0


@pytest.mark.unit
def test_detotalize_rollover_is_the_counters_modulus():
    # A 6-digit counter: rollover=1000000, as the reference docs say.
    sf = _frame({"t": [999998.0, 999999.0, 0.0, 1.0]})
    out = _run(fs.clean.Detotalize("t", rollover=1000000), sf)
    assert out.values["t"].tolist()[1:] == [1.0, 1.0, 1.0]


# -- convert_units keeps the flags --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "cleaner",
    [fs.clean.Stuck("flow", min_duration="3h"), fs.clean.Range("flow", max=50)],
    ids=["stuck", "range"],
)
def test_convert_units_keeps_the_flags_set_before_it(cleaner):
    pytest.importorskip("pint")
    flow = np.r_[np.arange(10.0, 20.0), np.full(4, 7.0), 99.0, np.arange(20.0, 29.0)]
    sf = _frame({"flow": flow}, units={"flow": "m^3/hr"})
    cleaned = _run(cleaner, sf)
    assert (cleaned.flags["flow"] != 0).any()
    pipe = fs.Pipeline([cleaner, fs.select.ConvertUnits("flow", to="L/s")], name="u")
    out = fs.run(pipe.spec, sf).signal
    assert out.flags.equals(cleaned.flags)
    assert out.units["flow"] == "L/s"
    assert np.allclose(out.values["flow"], cleaned.values["flow"] / 3.6, equal_nan=True)


# -- knn / iterative: nothing to learn from ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "step",
    [fs.impute.KNN(["a", "b"]), fs.impute.Iterative(["a", "b"])],
    ids=["knn", "iterative"],
)
def test_a_matrix_imputer_leaves_all_empty_streams_unchanged(step):
    sf = _frame({"a": np.full(24, np.nan), "b": np.full(24, np.nan)})
    out = _run(step, sf)
    assert out.values.isna().all().all()
    assert out.flags.equals(sf.flags)


# -- the shared helpers --------------------------------------------------------------


@pytest.mark.unit
def test_whole_step_errors_name_the_field():
    sf = _frame({"y": np.arange(24.0), "x": np.arange(24.0)})
    cases = [
        (fs.select.Shift("y", by="7min"), "by"),
        (fs.impute.Regression("y", features=["x"], lags=["7min"]), "lags"),
        (fs.forecast.Persistence("y", horizon="90min"), "horizon"),
        (fs.forecast.SeasonalNaive("y", horizon="1h", season="90min"), "season"),
        (fs.forecast.Regression("y", horizon="1h", lags=["90min"]), "lags"),
    ]
    for step, field in cases:
        with pytest.raises(SignalConfigError, match="whole number of the data's") as e:
            fs.Pipeline([step], name="w").fit(sf)
        assert e.value.field == field
    off_grid = fs.forecast.Persistence(
        "y", horizon="1h", issue_time="2025-01-01T05:30:00Z"
    )
    with pytest.raises(SignalConfigError, match="isn't on the data's") as e:
        fs.Pipeline([off_grid], name="w").fit(sf)
    assert e.value.field == "issue_time"


@pytest.mark.unit
def test_a_lone_string_is_a_list_everywhere():
    assert fs.forecast.Regression("y", horizon="1h", exog="p", lags="1h").exog == ("p",)
    assert fs.forecast.Regression("y", horizon="1h", lags="1h").lags == ("1h",)
    assert fs.impute.KNN("y", features="x").features == ("x",)
    assert fs.correct.Calibrate(target="A", reference="B", features="x").features == (
        "x",
    )
    assert fs.select.Mask(keep_when="x > 0", streams="y").streams == ("y",)


@pytest.mark.unit
def test_an_unknown_registered_method_is_an_attribute_error():
    for module in (fs.clean, fs.impute, fs.forecast):
        with pytest.raises(AttributeError, match="no attribute 'not_a_method'"):
            module.not_a_method  # noqa: B018
        with pytest.raises(AttributeError):
            module._private  # noqa: B018
