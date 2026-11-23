# Apply data to a flex-pse model

This guide cleans a day of pump flow data, lines it up with a flex-pse model's hourly steps, fixes the pump's flow
from it, and solves the model. It needs flex-pse and HiGHS (see [Installation](../getting_started/installation.md)).
See [Handing data to flex-pse models](../explanation/flexpse_integration.md) for how it works.

## 1. A model

Any flex-pse flowsheet spec works. This one has a tank feeding a pump that uses 0.4 kWh per m³, a day of hourly
steps, and a flat electricity price:

```python
import copy
import pyomo.environ as pyo
from flexops import build_model

import flexsignal as fs
from flexsignal import handoff
from flexsignal.integrations import flexpse as fsp
from flexsignal.testing import synthetic

SPEC = {
    "schema_version": "0.1.1",
    "name": "site",
    "elements": [
        {"kind": "time", "start_date": "2025-01-07", "end_date": "2025-01-08", "time_step": "1 hr"},
        {"kind": "property_package", "name": "properties", "property_class": "SimpleAqueousFlow"},
        {"kind": "costing", "name": "costing",
         "energy_prices": {"electrical": {"value": 0.15, "units": "USD/kWh"}}},
        {"kind": "plant", "name": "site"},
        {"kind": "unit", "name": "site.tank", "unit_model_class": "Tank", "costing_package": None,
         "construction_options": {"max_volume": {"value": 2000.0, "units": "m^3"},
                                  "initial_volume": {"value": 1000.0, "units": "m^3"}},
         "connections": [{"port": "outlet", "to": "site.pump.inlet", "name": "arc_0"}]},
        {"kind": "unit", "name": "site.pump", "unit_model_class": "ConstantEnergyIntensityModel",
         "construction_options": {"energy_intensity": {"value": 0.4, "units": "kWh/m^3"}}},
        {"kind": "objective", "name": "objective"},
    ],
}
print(fsp.time_grid(SPEC)[:3])   # the model's first time points, naive local time
```

## 2. Clean the data

Clean with flex-signal first, so the hand-off only lines the data up:

```python
case = synthetic.plant(seed=4).with_spikes("flow", count=6).with_gaps("flow", count=4, length=2)
pipe = fs.Pipeline(name="flow_cleanup")
pipe.add_step(fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0))
pipe.add_step(fs.impute.Interpolate("flow", max_gap="1h"))
clean = fs.run(pipe.spec, case.signal()).signal
```

## 3. Line it up with the model's steps

```python
options = handoff.GridOptions(timezone="UTC", gaps="error")   # set your plant's zone
values = handoff.to_grid(clean, fsp.time_grid(SPEC), options, columns=["flow"])
print(len(values.values["flow"]), values.units["flow"], values.estimated)
```

`gaps="error"` refuses to fill anything silently: the cleaning did that already, and flagged it. `estimated` counts
the hourly steps that rest on filled samples, and `resampled` those that rest on samples put on a new grid (after
upsampling with `align`, most of those are interpolated between readings).

## 4. Fix the variables and solve

`apply_values` takes a spec or a built model, with the same result. On a spec, it returns a copy to build:

```python
overrides = values.to_overrides(rename={"flow": "site.pump.flow_in"}, strict=True)
model = build_model(fsp.apply_values(SPEC, overrides))
```

On a built model, it fixes the variables in place, which suits re-solving one model with each new day of data:

```python
model = build_model(copy.deepcopy(SPEC))
fsp.apply_values(model, overrides)
result = pyo.SolverFactory("appsi_highs").solve(model)
print(result.solver.termination_condition)
assert str(result.solver.termination_condition) == "optimal"
```

A price works the same way. On the spec, `fsp.apply_values(SPEC, {"costing.energy_prices.electrical": {"value":
0.2, "units": "USD/kWh"}})` sets the flat price. flex-pse builds that price as a constant, so on a built model only a
price the model holds as a mutable Param can change: `fsp.apply_values(model, {"price": [...]})`.

## Variations

- **Hourly energy onto 15-minute steps:** `GridOptions(resample={"agg": "sum"})` splits each hour's amount over its
  four steps.
- **Last year's data for this year's model:** `GridOptions(offset="-365D")`, or `data_years={2026: 2025}` to keep
  calendar dates.
- **One number, not a series:** `GridOptions(reduce="mean")`.
