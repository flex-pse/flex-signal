"""The extension API: registered cleaners, imputers, forecasters and models."""

import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.backends.models import unregister_model
from flexsignal.core import Flag, SignalConfigError, SignalDataError
from flexsignal.core.registry import unregister_step

PLUGIN = "flexsignal.steps.tests.plugin_methods"


@pytest.fixture
def plugin():
    """Load the test plugin module, and forget its methods afterwards."""
    sys.modules.pop(PLUGIN, None)
    names = fs.load_plugins(PLUGIN)
    yield names
    for name in names:
        unregister_step(name)
        unregister_model(name)
    sys.modules.pop(PLUGIN, None)


@pytest.fixture
def forget():
    """Unregister the names a test registers."""
    names = []
    yield names
    for name in names:
        unregister_step(name)
        unregister_model(name)


def _data(days=10, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="1h", tz="UTC")
    hours = np.arange(len(index))
    pressure = (
        5 + np.sin(2 * np.pi * hours / 24) + 0.05 * rng.standard_normal(len(index))
    )
    flow = 20 * pressure + 0.2 * rng.standard_normal(len(index))
    if days >= 7:
        flow[[30, 90, 150]] = [400.0, -300.0, 500.0]  # spikes
        flow[60:66] = np.nan  # a gap
    df = pd.DataFrame({"flow": flow, "pressure": pressure}, index=index)
    return fs.SignalFrame.from_frame(df, units={"flow": "m^3/hr", "pressure": "bar"})


def _python_pipeline():
    return fs.Pipeline(
        [
            fs.clean.TestDespike("flow", k=6.0),
            fs.impute.TestNeighbourFill("flow", features=["pressure"], max_gap="12h"),
            fs.forecast.TestDrift("flow", horizon="6h", quantiles=[0.1, 0.9]),
        ],
        name="registered",
    )


SPEC = {
    "schema_version": "0.1.0",
    "name": "registered",
    "requires": [PLUGIN],
    "steps": [
        {"kind": "TestDespike", "streams": ["flow"], "k": 6.0},
        {
            "kind": "TestNeighbourFill",
            "streams": ["flow"],
            "features": ["pressure"],
            "max_gap": "12h",
        },
        {
            "kind": "TestDrift",
            "streams": ["flow"],
            "horizon": "6h",
            "quantiles": [0.1, 0.9],
        },
    ],
}


# -- acceptance --------------------------------------------------------------------


@pytest.mark.component
def test_registered_methods_run_from_json_like_python_and_save(plugin, tmp_path):
    assert set(plugin) == {
        "TestDespike",
        "TestNeighbourFill",
        "TestDrift",
        "TestScaledOLS",
    }
    sf = _data()
    from_python = fs.run(_python_pipeline().spec, sf)
    from_json = fs.run(json.dumps(SPEC), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    assert fs.load_spec(fs.dump_spec(from_json.fitted.spec)) == from_json.fitted.spec

    out = from_json.signal
    assert out.has_flag(Flag.OUTLIER)["flow"].sum() == 3
    filled = out.has_flag(Flag.IMPUTED_MODEL)["flow"]
    assert filled.sum() == 9  # the gap, and the three spikes removed
    truth = 20 * out.values["pressure"][filled]
    assert np.allclose(out.values["flow"][filled], truth, rtol=0.02)
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 6
    assert (out.values["flow__q90"][ahead] > out.values["flow"][ahead]).all()

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved, plugins=[PLUGIN])
    again = loaded.transform(sf)
    pd.testing.assert_frame_equal(again.values, out.values)
    state = json.loads((saved / "fitted" / "00_TestDespike.json").read_text())
    assert set(state["streams"]["flow"]) == {"median", "spread"}


@pytest.mark.unit
def test_provenance_names_the_function(plugin):
    result = fs.run(_python_pipeline().spec, _data())
    record = result.provenance[0]
    assert record["function"] == {
        "module": PLUGIN,
        "function": "despike",
        "package": "flex-signal",
        "version": fs.__version__,
    }
    assert (
        "function"
        not in fs.run(
            fs.Pipeline([fs.clean.Range("flow", min=0)]).spec, _data()
        ).provenance[0]
    )


@pytest.mark.unit
def test_a_registered_model_is_a_model_choice(plugin):
    sf = _data()
    step = fs.impute.Regression("flow", features=["pressure"], model="TestScaledOLS")
    state = step.fit(sf)
    json.dumps(state)
    ols = fs.impute.Regression(
        "flow", features=["pressure"], model="sklearn:LinearRegression"
    )
    expected = ols.fit(sf)["models"]["flow"]
    assert state["models"]["flow"]["coef"] == pytest.approx(expected["coef"])
    assert state["models"]["flow"]["intercept"] == pytest.approx(expected["intercept"])
    fs.forecast.Regression(
        "flow", horizon="6h", fourier=[{"period": "1D"}], model="TestScaledOLS"
    )
    fs.correct.Calibrate(target="flow", reference="pressure", model="TestScaledOLS")


@pytest.mark.component
def test_a_registered_imputer_is_a_tier(plugin):
    sf = _data()
    step = fs.impute.Tiered(
        "flow",
        tiers=[
            {"max_gap": "2h", "method": "interpolate"},
            {
                "max_gap": None,
                "method": "TestNeighbourFill",
                "options": {"features": ["pressure"]},
            },
        ],
    )
    out = fs.run(fs.Pipeline([step]).spec, sf).signal
    assert out.has_flag(Flag.IMPUTED_MODEL)["flow"].sum() == 6
    with pytest.raises(SignalConfigError, match="unknown tier method 'Nope'"):
        fs.load_spec(
            {
                "schema_version": "0.1.0",
                "name": "x",
                "steps": [
                    {
                        "kind": "tiered",
                        "streams": ["flow"],
                        "tiers": [{"max_gap": None, "method": "Nope"}],
                    }
                ],
            }
        )


@pytest.mark.component
def test_a_registered_forecaster_backtests(plugin):
    sf = _data(days=6)
    spec = fs.Pipeline(
        [fs.forecast.TestDrift("pressure", horizon="6h", quantiles=[0.1, 0.9])]
    ).spec
    result = fs.backtest(
        spec,
        sf,
        folds=2,
        compare=[fs.forecast.Persistence("pressure", horizon="6h")],
    )
    assert set(result.summary()["method"]) == {"TestDrift", "forecast_persistence"}


@pytest.mark.unit
def test_registered_steps_pickle_by_kind(plugin):
    step = fs.clean.TestDespike("flow", k=3.0)
    assert pickle.loads(pickle.dumps(step)) == step


# -- what is refused ---------------------------------------------------------------


@pytest.mark.unit
def test_an_unregistered_kind_says_register_it_first():
    spec = {**SPEC, "requires": ["myplant.methods"]}
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(spec)
    assert err.value.field == "steps.0.kind"
    assert "register it first" in str(err.value)
    assert "fs.register_imputer('TestDespike', fn)" in str(err.value)
    assert "['myplant.methods']" in str(err.value)


@pytest.mark.unit
def test_a_spec_cannot_make_flexsignal_import_a_module(tmp_path, monkeypatch):
    (tmp_path / "planted_module.py").write_text("raise RuntimeError('imported!')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    attempts = [
        {**SPEC, "requires": ["planted_module"]},
        {**SPEC, "steps": [{"kind": "planted_module:fn", "streams": ["flow"]}]},
        {**SPEC, "steps": [{"kind": "planted_module", "streams": ["flow"]}]},
        {
            **SPEC,
            "requires": [],
            "steps": [
                {
                    "kind": "regression",
                    "streams": ["flow"],
                    "features": ["pressure"],
                    "model": "planted_module:Model",
                }
            ],
        },
    ]
    for spec in attempts:
        with pytest.raises(SignalConfigError):
            fs.load_spec(spec)
    assert "planted_module" not in sys.modules


@pytest.mark.unit
def test_invalid_options_fail_on_load_with_the_field_named(plugin):
    spec = {**SPEC, "steps": [{"kind": "TestDespike", "streams": ["flow"], "k": "x"}]}
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(spec)
    assert err.value.field == "steps.0.k"
    spec = {
        **SPEC,
        "steps": [{"kind": "TestNeighbourFill", "streams": ["flow"], "weight": 2}],
    }
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(spec)
    assert err.value.field == "steps.0.features"
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec({**SPEC, "steps": [{**SPEC["steps"][0], "window": "1h"}]})
    assert err.value.field == "steps.0.window"


@pytest.mark.unit
def test_signatures_that_cannot_be_options_are_refused(forget):
    def ok(values, *, k: float = 1.0):
        return values > k

    def no_hint(values, *, k=1.0):
        return values > k

    def var_kwargs(values, **options):
        return values > 0

    def reserved(values, *, streams: list = ()):
        return values > 0

    def too_many(values, state, extra, *, k: float = 1.0):
        return values > k

    def not_json(values, *, k: object = object()):  # noqa: B008
        return values > 0

    cases = [
        (no_hint, "type hint"),
        (var_kwargs, "keyword-only"),
        (reserved, "set by flex-signal"),
        (too_many, r"takes \(values, state, extra\)"),
        (not_json, "isn't JSON"),
    ]
    for fn, message in cases:
        with pytest.raises(SignalConfigError, match=message):
            fs.register_cleaner("BadCleaner", fn)
    with pytest.raises(SignalConfigError, match="must take the fitted state"):
        fs.register_cleaner("BadCleaner", ok, fit=lambda values: {})
    with pytest.raises(SignalConfigError, match="flag must be one of"):
        fs.register_cleaner("BadCleaner", ok, flag="IMPUTED_MODEL")
    for name in ("Range", "range", "not-a-name", "class"):
        with pytest.raises(SignalConfigError):
            fs.register_cleaner(name, ok)


@pytest.mark.unit
def test_registering_again_replaces_only_the_same_function(forget):
    def first(values, *, k: float = 1.0):
        return values > k

    forget.append("AgainCleaner")
    fs.register_cleaner("AgainCleaner", first)

    def first(values, *, k: float = 2.0):  # noqa: F811  (a re-run notebook cell)
        return values > k

    fs.register_cleaner("AgainCleaner", first)
    assert fs.clean.AgainCleaner("flow").k == 2.0

    def other(values, *, k: float = 1.0):
        return values > k

    with pytest.raises(SignalConfigError, match="already registered by"):
        fs.register_cleaner("AgainCleaner", other)
    with pytest.raises(SignalConfigError, match="already registered"):
        fs.register_imputer("AgainCleaner", lambda values, gaps: values)


@pytest.mark.unit
def test_bad_results_are_explained(forget):
    sf = _data(days=2)

    def wrong_length(values):
        return np.zeros(3, dtype=bool)

    def numbers(values):
        return values * 2.0

    def mean_only(history, future):
        return np.zeros(len(future))

    forget.extend(["WrongLength", "Numbers", "MeanOnly"])
    fs.register_cleaner("WrongLength", wrong_length)
    fs.register_cleaner("Numbers", numbers)
    fs.register_forecaster("MeanOnly", mean_only)
    with pytest.raises(SignalDataError, match="one value per row"):
        fs.clean.WrongLength("flow").transform(sf, {})
    with pytest.raises(SignalDataError, match="True/False"):
        fs.clean.Numbers("flow").transform(sf, {})
    step = fs.forecast.MeanOnly("pressure", horizon="2h", quantiles=[0.1])
    with pytest.raises(SignalDataError, match="only the mean"):
        step.transform(sf, step.fit(sf))
    plain = fs.forecast.MeanOnly("pressure", horizon="2h")
    out = plain.transform(sf, plain.fit(sf))
    assert out.has_flag(Flag.FORECAST)["pressure"].sum() == 2


@pytest.mark.unit
def test_unknown_names_and_plugins():
    with pytest.raises(AttributeError):
        fs.clean.NoSuchCleaner  # noqa: B018
    with pytest.raises(SignalConfigError, match="could not be imported"):
        fs.load_plugins("no_such_package.methods")
    with pytest.raises(SignalConfigError, match="isn't a module name"):
        fs.load_plugins("a/b")


@pytest.mark.unit
def test_run_loads_plugins(plugin):
    for name in plugin:
        unregister_step(name)
        unregister_model(name)
    sys.modules.pop(PLUGIN, None)
    result = fs.run(SPEC, _data(), plugins=[PLUGIN])
    assert result.signal.has_flag(Flag.FORECAST)["flow"].sum() == 6
