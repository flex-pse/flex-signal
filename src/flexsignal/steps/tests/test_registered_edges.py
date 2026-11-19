"""Extension-API edge cases: refused registrations and every result shape."""

import sys

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.backends.models import unregister_model
from flexsignal.core import Flag, SignalConfigError, SignalDataError
from flexsignal.core.registry import unregister_step
from flexsignal.steps.registered import registered_class, registered_methods

PLUGIN = "flexsignal.steps.tests.plugin_methods"


@pytest.fixture
def forget():
    """Unregister the names a test registers."""
    names = []
    yield names
    for name in names:
        unregister_step(name)
        unregister_model(name)


def _data(hours=48, gap=True):
    index = pd.date_range("2025-01-01", periods=hours, freq="1h", tz="UTC")
    t = np.arange(hours, dtype=float)
    flow = 10.0 + np.sin(2 * np.pi * t / 24)
    if gap:
        flow[5:7] = np.nan
    df = pd.DataFrame({"flow": flow, "pressure": 2.0 + 0.01 * t}, index=index)
    return fs.SignalFrame.from_frame(df)


# -- registration: what is refused ---------------------------------------------------


@pytest.mark.unit
def test_the_same_function_cannot_take_a_second_role(forget):
    def two_args(values, other):
        return values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values > 0

    forget.append("EdgeRoleClash")
    fs.register_cleaner("EdgeRoleClash", two_args)
    with pytest.raises(SignalConfigError, match="already registered as a cleaner"):
        fs.register_imputer("EdgeRoleClash", two_args)


@pytest.mark.unit
def test_things_that_are_not_functions_are_refused(forget):
    with pytest.raises(SignalConfigError, match="is not a function") as err:
        fs.register_cleaner("EdgeNotAFunction", 5)
    assert err.value.value == "EdgeNotAFunction"

    def with_state(values, state, *, k: float = 1.0):
        return values > k

    with pytest.raises(SignalConfigError, match="fit must be a function, not 5"):
        fs.register_cleaner("EdgeBadFit", with_state, fit=5)
    assert "EdgeNotAFunction" not in registered_methods()
    assert "EdgeBadFit" not in registered_methods()


@pytest.mark.unit
def test_unresolvable_type_hints_are_explained():
    def unresolved(values, *, k=1.0):
        return values > k

    unresolved.__annotations__ = {"k": "NoSuchType"}
    with pytest.raises(SignalConfigError, match="can't read the function's signature"):
        fs.register_cleaner("EdgeUnresolved", unresolved)


@pytest.mark.unit
def test_cleaner_flags_by_name_or_member(forget):
    def high(values, *, k: float = 10.0):
        return values > k

    with pytest.raises(SignalConfigError, match="flag must be one of") as err:
        fs.register_cleaner("EdgeFlag", high, flag="NOPE")
    assert err.value.field == "flag"
    forget.append("EdgeFlag")
    fs.register_cleaner("EdgeFlag", high, flag=Flag.STUCK)
    assert registered_methods()["EdgeFlag"].flag is Flag.STUCK
    out = fs.clean.EdgeFlag("flow").transform(_data(), {})
    assert out.has_flag(Flag.STUCK)["flow"].sum() == (_data().values["flow"] > 10).sum()


@pytest.mark.unit
def test_a_function_outside_any_package_has_no_package_in_its_source(forget):
    def loose(values, *, k: float = 1.0):
        return values > k

    loose.__module__ = "edge_loose_module"
    forget.append("EdgeLoose")
    fs.register_cleaner("EdgeLoose", loose)
    assert registered_methods()["EdgeLoose"].source == {
        "module": "edge_loose_module",
        "function": loose.__qualname__,
        "package": None,
        "version": None,
    }


@pytest.mark.unit
def test_private_names_are_never_registered_classes():
    assert registered_class("cleaner", "_hidden") is None


# -- fit -----------------------------------------------------------------------------


@pytest.mark.unit
def test_a_fit_with_var_kwargs_gets_every_option(forget):
    def above(values, state, *, k: float = 1.0):
        return values > state["level"] + k

    def fit_above(values, **options):
        return {"level": float(values.median()), "seen": sorted(options)}

    forget.append("EdgeKwargsFit")
    fs.register_cleaner("EdgeKwargsFit", above, fit=fit_above)
    assert registered_methods()["EdgeKwargsFit"].fit_options is None
    step = fs.clean.EdgeKwargsFit("flow", k=0.5)
    state = step.fit(_data())
    assert state["streams"]["flow"]["seen"] == ["k"]
    out = fs.run(fs.Pipeline([step]).spec, _data()).signal
    assert out.has_flag(Flag.OUTLIER)["flow"].sum() > 0


@pytest.mark.unit
def test_a_fit_that_returns_no_dict_is_refused(forget):
    def above(values, state):
        return values > 0

    forget.append("EdgeListFit")
    fs.register_cleaner("EdgeListFit", above, fit=lambda values: [1.0])
    with pytest.raises(SignalConfigError, match="fit returned list"):
        fs.clean.EdgeListFit("flow").fit(_data())


# -- what a cleaner may return -------------------------------------------------------


@pytest.fixture
def shapes(forget):
    """A cleaner that returns its mask in the shape its ``mode`` option names."""

    def shaped(values, *, mode: str = "bool"):
        mask = values > 10.0
        if mode == "frame":
            return mask.to_frame()
        if mode == "two_columns":
            return pd.concat([mask, mask], axis=1)
        if mode == "plain_index":
            return mask.reset_index(drop=True)
        if mode == "shifted":
            return mask.set_axis(mask.index + pd.Timedelta("1min"))
        if mode == "ints":
            return mask.astype(int)
        if mode == "text":
            return np.array(["yes"] * len(values))
        if mode == "fractions":
            return np.full(len(values), 0.5)
        return mask

    forget.append("EdgeShapes")
    fs.register_cleaner("EdgeShapes", shaped)
    return fs.clean.EdgeShapes


@pytest.mark.unit
def test_cleaner_results_that_can_be_read_as_masks(shapes):
    sf = _data()
    expected = shapes("flow").transform(sf, {})
    assert shapes("flow").fit(sf) == {}
    assert expected.has_flag(Flag.OUTLIER)["flow"].sum() > 0
    for mode in ("frame", "plain_index", "ints"):
        assert shapes("flow", mode=mode).transform(sf, {}).equals(expected), mode


@pytest.mark.unit
def test_cleaner_results_that_cannot_be_masks(shapes):
    sf = _data()
    cases = [
        ("two_columns", "a table with 2 columns"),
        ("shifted", "different time index"),
        ("text", "must return True/False"),
        ("fractions", "must return True/False"),
    ]
    for mode, message in cases:
        with pytest.raises(SignalDataError, match=message):
            shapes("flow", mode=mode).transform(sf, {})


# -- imputers ------------------------------------------------------------------------


@pytest.mark.unit
def test_an_imputer_without_state_fills_only_streams_with_gaps(forget):
    seen = []

    def constant(values, gaps, *, fill: float = 7.0, text: bool = False):
        seen.append(gaps.name)
        if text:
            return np.array(["seven"] * len(values))
        return np.full(len(values), fill)

    forget.append("EdgeConstant")
    fs.register_imputer("EdgeConstant", constant)
    sf = _data()
    step = fs.impute.EdgeConstant(["flow", "pressure"])
    assert step.fit(sf) == {}
    out = step.transform(sf, {})
    assert seen == ["flow"]  # pressure has no gaps, so the function isn't called
    assert out.values["flow"].iloc[5:7].tolist() == [7.0, 7.0]
    assert out.has_flag(Flag.IMPUTED_MODEL)["flow"].sum() == 2
    with pytest.raises(SignalDataError, match="must return numbers"):
        fs.impute.EdgeConstant("flow", text=True).transform(sf, {})


# -- forecasters ---------------------------------------------------------------------


@pytest.fixture
def levels(forget):
    """A forecaster with a fit, whose result shape its ``mode`` option names."""

    def level(history, future, state, *, mode: str = "mean"):
        n = len(future)
        mean = np.full(n, state["level"])
        if mode == "no_mean":
            return {"value": mean}
        if mode == "named":
            return {"mean": mean, "q10": mean - 1.0, 0.9: mean + 1.0}
        if mode == "frame":
            return pd.DataFrame({"mean": mean}, index=future.index)
        return {"mean": mean}

    def fit_level(history, *, mode: str = "mean"):
        return {"level": float(history.iloc[:, 0].mean())}

    forget.append("EdgeLevel")
    fs.register_forecaster("EdgeLevel", level, fit=fit_level)
    return fs.forecast.EdgeLevel


@pytest.mark.unit
def test_forecaster_results_mean_only_or_with_named_quantiles(levels):
    sf = _data(gap=False)
    level = float(sf.values["pressure"].mean())
    for mode in ("mean", "frame"):
        step = levels("pressure", horizon="2h", mode=mode)
        out = step.transform(sf, step.fit(sf))
        ahead = out.has_flag(Flag.FORECAST)["pressure"]
        assert ahead.sum() == 2
        assert np.allclose(out.values["pressure"][ahead], level), mode
    step = levels("pressure", horizon="2h", quantiles=[0.1, 0.9], mode="named")
    out = step.transform(sf, step.fit(sf))
    ahead = out.has_flag(Flag.FORECAST)["pressure"]
    assert np.allclose(out.values["pressure__q10"][ahead], level - 1.0)
    assert np.allclose(out.values["pressure__q90"][ahead], level + 1.0)


@pytest.mark.unit
def test_forecaster_results_that_are_refused(levels):
    sf = _data(gap=False)
    step = levels("pressure", horizon="2h", mode="no_mean")
    with pytest.raises(SignalDataError, match="needs a 'mean' entry"):
        step.transform(sf, step.fit(sf))
    step = levels("pressure", horizon="2h", quantiles=[0.1], mode="mean")
    with pytest.raises(SignalDataError, match=r"no \['q10'\] and no 'std'"):
        step.transform(sf, step.fit(sf))


# -- plugins -------------------------------------------------------------------------


@pytest.mark.unit
def test_load_plugins_takes_lists_of_names():
    sys.modules.pop(PLUGIN, None)
    names = fs.load_plugins([PLUGIN])
    try:
        assert "TestDespike" in names
    finally:
        for name in names:
            unregister_step(name)
            unregister_model(name)
        sys.modules.pop(PLUGIN, None)
    with pytest.raises(SignalConfigError, match="plugins are module names, not 5"):
        fs.load_plugins(5)
