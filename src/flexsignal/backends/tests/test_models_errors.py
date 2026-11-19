import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.backends.models import (
    fit_curve,
    fit_linear,
    registered_model_class,
    unregister_model,
)
from flexsignal.core import SignalConfigError, SignalDataError


class _Mean:
    """A regressor that predicts the mean of y (coef_ all zero)."""

    def fit(self, X, y):
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = float(np.mean(y))
        return self


class _OtherMean(_Mean):
    pass


class _NoCoef:
    def fit(self, X, y):
        return self


class _NoFit:
    pass


@pytest.fixture
def forget():
    """Unregister the model names a test registers."""
    names = []
    yield names
    for name in names:
        unregister_model(name)


def _line(n=12):
    X = np.arange(float(n)).reshape(-1, 1)
    return X, 3.0 + 2.0 * X[:, 0]


@pytest.mark.unit
def test_register_model_refuses_a_name_that_is_not_an_identifier():
    with pytest.raises(SignalConfigError, match="must be a Python identifier") as exc:
        fs.register_model("plant-ridge", _Mean)
    assert exc.value.value == "plant-ridge"
    assert registered_model_class("plant-ridge") is None


@pytest.mark.unit
def test_register_model_refuses_a_class_without_fit():
    with pytest.raises(SignalConfigError, match=r"has no fit\(X, y\) method") as exc:
        fs.register_model("NoFitModel", _NoFit)
    assert exc.value.value == "NoFitModel"
    assert registered_model_class("NoFitModel") is None


@pytest.mark.unit
def test_register_model_refuses_a_name_taken_by_another_class(forget):
    forget.append(fs.register_model("TakenModel", _Mean))
    # The same class again is fine (a re-run notebook cell).
    assert fs.register_model("TakenModel", _Mean) == "TakenModel"
    with pytest.raises(SignalConfigError, match="already registered for") as exc:
        fs.register_model("TakenModel", _OtherMean)
    assert "_Mean" in exc.value.message and exc.value.value == "TakenModel"
    assert registered_model_class("TakenModel") is _Mean


@pytest.mark.unit
def test_registered_model_class_returns_the_class_or_none(forget):
    assert registered_model_class("MeanModel") is None
    forget.append(fs.register_model("MeanModel", _Mean, description="mean"))
    assert registered_model_class("MeanModel") is _Mean
    X, y = _line()
    state = fit_linear("MeanModel", X, y)
    assert state == {"intercept": pytest.approx(np.mean(y)), "coef": [0.0]}


@pytest.mark.unit
def test_fit_linear_needs_more_rows_than_coefficients():
    X = np.ones((2, 2))
    with pytest.raises(SignalDataError, match="only 2 rows to fit 3 coefficients"):
        fit_linear("sklearn:LinearRegression", X, np.ones(2))


@pytest.mark.unit
def test_registered_model_without_coefficients_is_refused(forget):
    forget.append(fs.register_model("NoCoefModel", _NoCoef))
    X, y = _line()
    with pytest.raises(SignalConfigError, match="has no coef_ and intercept_"):
        fit_linear("NoCoefModel", X, y)


@pytest.mark.unit
def test_theil_sen_is_seeded_so_fits_repeat():
    X, y = _line(30)
    first = fit_linear("sklearn:TheilSenRegressor", X, y)
    assert first == fit_linear("sklearn:TheilSenRegressor", X, y)
    assert first["coef"] == [pytest.approx(2.0)]
    assert first["intercept"] == pytest.approx(3.0)


@pytest.mark.unit
def test_params_the_model_does_not_accept_are_a_config_error():
    X, y = _line()
    with pytest.raises(SignalConfigError, match="doesn't accept params") as exc:
        fit_linear("sklearn:LinearRegression", X, y, {"bogus": 1})
    assert exc.value.field == "params"
    assert "['bogus']" in exc.value.message


@pytest.mark.unit
def test_fit_curve_needs_more_rows_than_parameters():
    df = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(SignalDataError, match="only 2 rows to fit 2 parameters"):
        fit_curve("a + b * A", df, np.array([1.0, 2.0]), {"a": 0.0, "b": 1.0})


@pytest.mark.unit
def test_fit_curve_that_does_not_converge_is_a_data_error():
    df = pd.DataFrame({"A": np.linspace(0.0, 1.0, 20)})
    y = 2.0 * np.exp(3.0 * df["A"].to_numpy())
    with pytest.raises(SignalDataError, match="didn't converge") as exc:
        fit_curve("a * exp(b * A)", df, y, {"a": 1.0, "b": 1.0}, {"maxfev": 2})
    assert "maxfev" in exc.value.message


@pytest.mark.unit
@pytest.mark.parametrize("key", ["statsmodels:OLS", "statsmodels:RLM"])
def test_statsmodels_refuses_an_unknown_param_instead_of_ignoring_it(key):
    X = np.arange(20.0)[:, None]
    y = 2.0 * X[:, 0] + 1.0
    with pytest.raises(SignalConfigError, match="bogus") as err:
        fit_linear(key, X, y, {"bogus": 1})
    assert err.value.field == "params"
    assert fit_linear(key, X, y, {"missing": "drop"})["coef"] == pytest.approx([2.0])
