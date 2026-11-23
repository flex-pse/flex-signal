"""flex-pse: a model's time grid, fixing variables from data, and the whole hand-off.

The spec-editing tests need no flex-pse; the rest are skipped without it.
"""

import copy
import sys
import types

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import SignalConfigError
from flexsignal.handoff import GridOptions, fit_data, split_holdout, to_grid
from flexsignal.integrations import flexpse as fsp
from flexsignal.testing import synthetic

SPEC = {
    "schema_version": "0.1.1",
    "name": "site",
    "elements": [
        {
            "kind": "time",
            "start_date": "2025-01-07",
            "end_date": "2025-01-08",
            "time_step": "1 hr",
        },
        {
            "kind": "property_package",
            "name": "properties",
            "property_class": "SimpleAqueousFlow",
        },
        {
            "kind": "costing",
            "name": "costing",
            "energy_prices": {"electrical": {"value": 0.15, "units": "USD/kWh"}},
        },
        {"kind": "plant", "name": "site"},
        {
            "kind": "unit",
            "name": "site.tank",
            "unit_model_class": "Tank",
            "costing_package": None,
            "construction_options": {
                "max_volume": {"value": 2000.0, "units": "m^3"},
                "initial_volume": {"value": 1000.0, "units": "m^3"},
            },
            "connections": [
                {"port": "outlet", "to": "site.pump.inlet", "name": "arc_0"}
            ],
        },
        {
            "kind": "unit",
            "name": "site.pump",
            "unit_model_class": "ConstantEnergyIntensityModel",
            "construction_options": {
                "energy_intensity": {"value": 0.4, "units": "kWh/m^3"}
            },
        },
        {"kind": "objective", "name": "objective"},
    ],
}
"""A small flowsheet: a tank feeding a pump that uses 0.4 kWh per m^3, a day of
hourly steps from 7 January 2025, and a flat electricity price."""


# -- spec editing: no flex-pse needed ------------------------------------------------


@pytest.mark.unit
def test_apply_values_to_a_spec_adds_named_fix_elements_without_mutating():
    original = copy.deepcopy(SPEC)
    spec = fsp.apply_values(
        SPEC,
        {
            "site.pump.flow_in": 10.0,
            "site.tank.flow_out": {"value": 1, "units": "m^3/s"},
        },
    )
    assert SPEC == original
    assert spec["elements"][-2:] == [
        {
            "kind": "fix",
            "name": "override:site.pump.flow_in",
            "variable": "site.pump.flow_in",
            "value": 10.0,
        },
        {
            "kind": "fix",
            "name": "override:site.tank.flow_out",
            "variable": "site.tank.flow_out",
            "value": 1,
            "units": "m^3/s",
        },
    ]


@pytest.mark.unit
def test_apply_values_to_a_spec_edits_costing_fields():
    before = copy.deepcopy(SPEC)
    price = {"value": 0.2, "units": "USD/kWh"}
    spec = fsp.apply_values(
        SPEC, {"costing.energy_prices.electrical": price, "site.pump.flow_in": 1.0}
    )
    assert SPEC == before
    costing = next(e for e in spec["elements"] if e.get("kind") == "costing")
    assert costing["energy_prices"]["electrical"] == price
    assert spec["elements"][-1]["variable"] == "site.pump.flow_in"
    with pytest.raises(SignalConfigError, match="'value' is .*not a dict"):
        fsp.apply_values(SPEC, {"costing.energy_prices.electrical.value.x": 1})


@pytest.mark.unit
def test_time_grid_refuses_design_specs_and_reads_no_time_as_none():
    assert fsp.time_grid({"elements": []}) is None
    design = {"base": {}, "years": [2030], "months": [1], "capacities": {}}
    with pytest.raises(SignalConfigError, match="not design specs"):
        fsp.time_grid(design)


@pytest.mark.unit
def test_apply_values_groups_paths_by_time_block(monkeypatch):
    applied = []
    fake = types.ModuleType("flexops")
    fake.apply_spec = lambda block, elements: applied.append((block, elements))
    monkeypatch.setitem(sys.modules, "flexops", fake)
    monkeypatch.setattr(fsp, "_need_flexpse", lambda what: None)
    blocks = {"plant_a": "block a", "plant_b": "block b"}
    monkeypatch.setattr(fsp, "time_scopes", lambda model: blocks)
    monkeypatch.setattr(fsp, "_find_params", lambda model, values: {})
    fsp.apply_values("model", {"plant_a.pump.flow": 1.0, "plant_b.pump.flow": [2.0]})
    assert [(block, [e["variable"] for e in els]) for block, els in applied] == [
        ("block a", ["pump.flow"]),
        ("block b", ["pump.flow"]),
    ]
    with pytest.raises(SignalConfigError, match="under none of them"):
        fsp.apply_values("model", {"pump.flow": 1.0})


# -- with flex-pse -----------------------------------------------------------------


@pytest.fixture
def flexpse():
    pytest.importorskip("flexops")
    import pyomo.environ as pyo
    from flexops import build_model

    return pyo, build_model


@pytest.mark.integration
def test_time_grid_matches_the_built_model(flexpse):
    pyo, build_model = flexpse
    grid = fsp.time_grid(SPEC)
    assert len(grid) == 24 and grid[0] == pd.Timestamp("2025-01-07")
    model = build_model(copy.deepcopy(SPEC))
    assert list(fsp.time_grid(model)) == list(grid)
    assert list(fsp.time_scopes(model)) == [""]
    bad = copy.deepcopy(SPEC)
    bad["elements"][0]["end_date"] = "2025-01-07T05:30"
    with pytest.raises(SignalConfigError, match="whole number of"):
        fsp.time_grid(bad)


@pytest.mark.integration
def test_a_spec_and_a_built_model_get_the_same_fixed_values(flexpse):
    pyo, build_model = flexpse
    overrides = {"site.pump.flow_in": [float(i) for i in range(24)]}
    from_spec = build_model(fsp.apply_values(SPEC, overrides))
    built = build_model(copy.deepcopy(SPEC))
    assert fsp.apply_values(built, overrides) is built
    for model in (from_spec, built):
        flow = model.site.pump.flow_in
        assert [pyo.value(flow[t]) for t in flow] == overrides["site.pump.flow_in"]
        assert all(flow[t].fixed for t in flow)


@pytest.mark.integration
def test_a_price_is_set_on_a_built_model_when_it_is_a_param(flexpse):
    pyo, build_model = flexpse
    model = build_model(copy.deepcopy(SPEC))
    with pytest.raises(SignalConfigError, match="built the spec's flat price"):
        fsp.apply_values(model, {"costing.energy_prices.electrical": 0.2})
    # A price the model holds as a mutable Param is set, in its own units.
    model.price = pyo.Param(
        model.site.pump.flow_in.index_set(),
        initialize=0.15,
        mutable=True,
        units=pyo.units.USD / pyo.units.kWh,
    )
    fsp.apply_values(model, {"price": {"value": 200.0, "units": "USD/MWh"}})
    assert [pyo.value(model.price[t]) for t in model.price] == pytest.approx([0.2] * 24)
    fsp.apply_values(model, {"price": [0.1] * 23 + [0.3]})
    assert pyo.value(model.price[model.price.index_set().last()]) == 0.3
    with pytest.raises(SignalConfigError, match="24 values to set, but 2"):
        fsp.apply_values(model, {"price": [0.1, 0.2]})


@pytest.mark.integration
def test_raw_data_to_a_solved_model_and_a_fitted_regressor(flexpse):
    """End to end: raw data -> clean -> the model's grid -> fixed -> solved."""
    pyo, build_model = flexpse
    pytest.importorskip("highspy")
    from flexparameterize import LinearRegressor

    # A week of 5-minute plant data with spikes and a frozen flow meter.
    case = (
        synthetic.plant(seed=4)
        .with_spikes("flow", count=6)
        .with_stuck("flow", length=24)
        .with_gaps("flow", count=4, length=2)
    )
    pipe = fs.Pipeline(name="flow_cleanup")
    pipe.add_step(fs.clean.Stuck("flow", min_duration="1h"))
    pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0))
    pipe.add_step(fs.impute.Interpolate("flow", max_gap="3h"))
    clean = fs.run(pipe.spec, case.signal()).signal

    model = build_model(copy.deepcopy(SPEC))
    grid = fsp.time_grid(model)
    values = to_grid(
        clean, grid, GridOptions(timezone="UTC", gaps="error"), columns=["flow"]
    )
    assert values.units == {"flow": "m^3/hr"}
    assert values.estimated["flow"] > 0  # some hours rest on filled samples
    fsp.apply_values(model, values.to_overrides(rename={"flow": "site.pump.flow_in"}))
    result = pyo.SolverFactory("appsi_highs").solve(model)
    assert str(result.solver.termination_condition) == "optimal"
    fixed = [pyo.value(model.site.pump.flow_in[t]) for t in model.site.pump.flow_in]
    assert np.allclose(fixed, values.values["flow"])
    # The hourly means of the clean flow are close to the true hourly flow.
    truth = case.clean["flow"].resample("1h").mean().loc["2025-01-07"].to_numpy()
    assert np.abs(np.array(fixed) - truth).max() < 2.0

    # Fit data for a regressor: pressure from flow, hourly, the last day held out.
    fit = fit_data(
        clean.select(["flow"]),
        clean.select(["pressure"]),
        {"resample": {"freq": "1h"}, "gaps": "drop"},
    )
    X_train, y_train, X_test, y_test = split_holdout(fit.X, fit.y, "1D")
    regressor = LinearRegressor().fit(
        X_train, y_train, input_units=fit.input_units, output_units=fit.output_units
    )
    coefficients = regressor.to_fit_result().coefficients
    # The synthetic plant's pressure is 4 + 0.01 x flow bar.
    assert coefficients["flow"] == pytest.approx(0.01, abs=0.002)
    assert coefficients["intercept"] == pytest.approx(4.0, abs=0.2)
    predicted = coefficients["intercept"] + coefficients["flow"] * X_test["flow"]
    scores = fs.eval.metrics(y_test["pressure"], predicted)
    assert scores["rmse"] < 0.05


@pytest.mark.integration
def test_a_forecast_is_fixed_into_a_model_and_solved(flexpse):
    """End to end: tomorrow's forecast flow → the model's grid → fixed → solved."""
    pyo, build_model = flexpse
    pytest.importorskip("highspy")
    # A week of data, 6-12 January; the model is the next day, 13 January.
    case = synthetic.plant(seed=4).with_spikes("flow", count=6)
    pipe = fs.Pipeline(name="flow_forecast")
    pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0))
    pipe.add_step(fs.select.Resample(freq="1h"))
    pipe.add_step(
        fs.forecast.Structural(
            "flow", horizon="1D", seasonal=[{"period": "1D", "harmonics": 2}]
        )
    )
    forecast = fs.run(pipe.spec, case.signal()).signal
    spec = copy.deepcopy(SPEC)
    spec["elements"][0].update(start_date="2025-01-13", end_date="2025-01-14")
    values = to_grid(
        forecast, fsp.time_grid(spec), GridOptions(timezone="UTC"), columns=["flow"]
    )
    assert values.estimated["flow"] == 24  # every hour rests on the forecast
    model = build_model(
        fsp.apply_values(
            spec, values.to_overrides(rename={"flow": "site.pump.flow_in"})
        )
    )
    result = pyo.SolverFactory("appsi_highs").solve(model)
    assert str(result.solver.termination_condition) == "optimal"
    fixed = [pyo.value(model.site.pump.flow_in[t]) for t in model.site.pump.flow_in]
    # The same plant one day longer gives the true flow on the model's day. With a
    # week of data the model has no weekly cycle, so it drifts a few m^3/h by night.
    day = synthetic.plant(seed=4, periods=8 * 288).underlying["flow"]
    truth = day.resample("1h").mean().loc["2025-01-13"].to_numpy()
    assert np.abs(np.array(fixed) - truth).max() < 5.0


@pytest.mark.unit
def test_without_flexpse_the_error_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "flexops", None)  # import flexops now fails
    with pytest.raises(fs.core.SignalBackendError, match=r"flex-signal\[flexpse\]"):
        fsp.time_grid(SPEC)
    with pytest.raises(fs.core.SignalBackendError, match="needs flex-pse"):
        fsp.apply_values(object(), {"site.pump.flow_in": 1.0})
