"""Regression tests for review findings in the backend registry and curve fits."""

import numpy as np
import pandas as pd
import pytest

from flexsignal.backends import get_backend, register_backend
from flexsignal.backends.models import fit_curve
from flexsignal.core import SignalBackendError, SignalConfigError


def _line():
    X = pd.DataFrame({"A": np.linspace(1.0, 10.0, 20)})
    return X, 2.0 + 3.0 * X["A"].to_numpy()


@pytest.mark.unit
def test_fit_curve_refuses_a_misspelled_param_as_a_config_error():
    X, y = _line()
    with pytest.raises(SignalConfigError, match="maxfevv") as exc:
        fit_curve("a + b * A", X, y, {"a": 0.0, "b": 1.0}, {"maxfevv": 10})
    assert exc.value.field == "params"


@pytest.mark.unit
def test_fit_curve_refuses_p0_in_params():
    X, y = _line()
    with pytest.raises(SignalConfigError, match="initial") as exc:
        fit_curve("a + b * A", X, y, {"a": 0.0, "b": 1.0}, {"p0": [0.0, 0.0]})
    assert exc.value.field == "params"


@pytest.mark.unit
def test_fit_curve_refuses_a_formula_that_is_nan_at_the_initial_values():
    X, y = _line()
    # log(a - A) is NaN for every row when a = 0; it used to "fit" and return a = 0.
    with pytest.raises(SignalConfigError, match="NaN or infinity") as exc:
        fit_curve("log(a - A)", X, y, {"a": 0.0})
    assert exc.value.field == "initial"


@pytest.mark.unit
def test_fit_curve_still_fits_a_good_formula():
    X, y = _line()
    fitted = fit_curve("a + b * A", X, y, {"a": 0.0, "b": 1.0}, {"maxfev": 5000})
    assert fitted == pytest.approx({"a": 2.0, "b": 3.0})


@pytest.mark.unit
def test_get_backend_takes_registered_keys_only():
    with pytest.raises(SignalBackendError, match="unknown backend key") as exc:
        get_backend("os:system")
    assert "scipy:PchipInterpolator" in exc.value.message


@pytest.mark.unit
def test_register_backend_needs_a_module_and_attribute_target():
    with pytest.raises(SignalBackendError, match="package.module:attribute"):
        register_backend("test:bad_target", "numpy.interp")
