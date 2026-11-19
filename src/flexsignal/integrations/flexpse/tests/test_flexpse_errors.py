"""flex-pse: the errors and edge cases of reading time grids and setting values.

Small hand-built pyomo models and specs; all skipped without flex-pse.
"""

import copy

import pytest

from flexsignal.core import SignalConfigError
from flexsignal.integrations import flexpse as fsp

pyo = TimeBlock = None  # set by the fixture below, when a test runs


@pytest.fixture(autouse=True)
def flexpse():
    """Import flex-pse when a test runs, or skip without it.

    Not at collection: importing flex-pse imports IDAES, and every pytest
    worker collects every file, so a module-level import would make each
    worker import IDAES at once, even with no flex-pse test to run.
    """
    global pyo, TimeBlock
    pytest.importorskip("flexops")
    import pyomo.environ as pyo
    from flexops.core.time_block import TimeBlock


TIME = {
    "kind": "time",
    "start_date": "2025-01-07",
    "end_date": "2025-01-08",
    "time_step": "1 hr",
}
"""A day of hourly steps from 7 January 2025."""


def spec_with_time(**changes):
    """Return a spec holding only a time element, with ``changes`` applied."""
    return {"elements": [{**copy.deepcopy(TIME), **changes}]}


def model_with_param(**options):
    """Return a model with a 3-point Param ``p`` indexed 0..2."""
    m = pyo.ConcreteModel()
    m.p = pyo.Param(range(3), initialize=1.0, **options)
    return m


# -- time grids ----------------------------------------------------------------------


@pytest.mark.integration
def test_time_grid_refuses_a_spec_with_two_time_elements():
    spec = {"elements": [copy.deepcopy(TIME), copy.deepcopy(TIME)]}
    with pytest.raises(SignalConfigError, match="more than one time element"):
        fsp.time_grid(spec)


@pytest.mark.integration
@pytest.mark.parametrize(
    "changes",
    [{"start_date": "not a date"}, {"time_step": "1 banana"}, {"time_step": None}],
)
def test_time_grid_refuses_an_unreadable_time_element(changes):
    with pytest.raises(SignalConfigError, match="Can't read the spec's time element"):
        fsp.time_grid(spec_with_time(**changes))


@pytest.mark.integration
def test_time_grid_refuses_dates_with_a_time_zone():
    spec = spec_with_time(end_date="2025-01-08T00:00+01:00")
    with pytest.raises(SignalConfigError, match="naive local time"):
        fsp.time_grid(spec)


@pytest.mark.integration
def test_time_grid_of_a_model_without_a_time_block_is_none():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    assert fsp.time_grid(m) is None
    assert fsp.time_scopes(m) == {}


@pytest.mark.integration
def test_time_grid_of_a_model_with_several_time_blocks_is_per_scope():
    m = pyo.ConcreteModel()
    m.plant_a = pyo.Block()
    m.plant_a.time_block = TimeBlock(
        start_date="2025-01-07", end_date="2025-01-08", time_step=pyo.units.hr
    )
    m.plant_b = pyo.Block()
    m.plant_b.time_block = TimeBlock(
        start_date="2025-01-07", end_date="2025-01-07T02:00", time_step=pyo.units.hr
    )
    grids = fsp.time_grid(m)
    assert sorted(grids) == ["plant_a", "plant_b"]
    assert len(grids["plant_a"]) == 24 and len(grids["plant_b"]) == 2


@pytest.mark.integration
def test_one_block_holding_two_time_blocks_is_refused():
    m = pyo.ConcreteModel()
    m.plant = pyo.Block()
    for name in ("first", "second"):
        block = TimeBlock(
            start_date="2025-01-07", end_date="2025-01-08", time_step=pyo.units.hr
        )
        setattr(m.plant, name, block)
    with pytest.raises(SignalConfigError, match="'plant' holds two time blocks"):
        fsp.time_scopes(m)
    with pytest.raises(SignalConfigError, match="holds two time blocks"):
        fsp.time_grid(m)


# -- setting Params ------------------------------------------------------------------


@pytest.mark.integration
def test_an_immutable_param_is_refused():
    m = model_with_param()
    with pytest.raises(SignalConfigError, match="'p': it is an immutable Param"):
        fsp.apply_values(m, {"p": 2.0})
    assert [pyo.value(m.p[i]) for i in m.p] == [1.0, 1.0, 1.0]


@pytest.mark.integration
def test_a_param_is_set_from_a_dict_of_index_values():
    m = model_with_param(mutable=True)
    fsp.apply_values(m, {"p": {0: 5.0, 2: 7.0}})
    assert [pyo.value(m.p[i]) for i in m.p] == [5.0, 1.0, 7.0]


@pytest.mark.integration
def test_a_dict_naming_a_missing_index_sets_nothing():
    m = model_with_param(mutable=True)
    with pytest.raises(SignalConfigError, match="'p' has no index 9"):
        fsp.apply_values(m, {"p": {0: 5.0, 9: 7.0}})
    assert [pyo.value(m.p[i]) for i in m.p] == [1.0, 1.0, 1.0]


@pytest.mark.integration
def test_a_param_value_with_units_is_converted_in_every_shape():
    m = model_with_param(mutable=True, units=pyo.units.kW)
    fsp.apply_values(m, {"p": {"value": [1.0, 2.0, 3.0], "units": "MW"}})
    assert [pyo.value(m.p[i]) for i in m.p] == pytest.approx([1e3, 2e3, 3e3])
    fsp.apply_values(m, {"p": {"value": {1: 0.5}, "units": "MW"}})
    assert [pyo.value(m.p[i]) for i in m.p] == pytest.approx([1e3, 500.0, 3e3])


@pytest.mark.integration
def test_a_param_without_units_takes_a_dimensionless_value():
    m = model_with_param(mutable=True)
    fsp.apply_values(m, {"p": {"value": 4.0, "units": "dimensionless"}})
    assert [pyo.value(m.p[i]) for i in m.p] == [4.0, 4.0, 4.0]
