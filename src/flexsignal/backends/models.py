"""Fit and apply the models that ``calibrate`` and ``regression`` use.

Every model here is linear in its inputs: ``y = intercept + X @ coef``. So the
fitted state is a handful of numbers that save as plain JSON, and applying a
fitted model needs only numpy, not the library that fitted it. The libraries
differ in *how* they estimate the coefficients, e.g. ordinary least squares or a
fit that down-weights outliers.

``scipy:curve_fit`` fits the parameters of a formula instead, for example
``"a + b * A + c * A ** 2"``. The formula is checked and evaluated by
:mod:`flexsignal.core.conditions`, so it can't run code.

Only the keys in :data:`LINEAR_MODELS` (and ``scipy:curve_fit``) are accepted. A
spec can't name an arbitrary ``"package.module:Class"`` here, because loading a
spec must never import code. The keys themselves are registered, with the other
built-in keys, in :mod:`flexsignal.backends.builtin`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from flexsignal.backends.registry import get_backend
from flexsignal.core.conditions import evaluate_formula
from flexsignal.core.exceptions import SignalConfigError, SignalDataError

LINEAR_MODELS: dict[str, str] = {
    "sklearn:LinearRegression": "Ordinary least squares.",
    "sklearn:Ridge": "Least squares with an L2 penalty (params: alpha).",
    "sklearn:HuberRegressor": "Robust fit that down-weights outliers (epsilon).",
    "sklearn:TheilSenRegressor": "Robust median-of-slopes fit; slower.",
    "statsmodels:OLS": "Ordinary least squares (statsmodels).",
    "statsmodels:RLM": "Robust linear model with Huber's T norm (statsmodels).",
}
"""Model keys accepted by ``calibrate`` and ``regression``, with one line each."""

CURVE_FIT = "scipy:curve_fit"
"""Key for fitting a formula's parameters with :func:`scipy.optimize.curve_fit`."""


_REGISTERED: dict[str, tuple[type, str]] = {}


def register_model(name: str, cls: type, *, description: str = "") -> str:
    """Make a regressor of your own a ``model`` for ``calibrate`` and ``regression``.

    The class must work like a scikit-learn regressor: ``cls(**params).fit(X, y)``,
    after which ``coef_`` and ``intercept_`` describe the fitted model,
    ``y = intercept_ + X @ coef_``. Only those numbers are kept, so the fitted
    state stays JSON and applying it needs only numpy.

    Args:
        name: The name a spec's ``model`` field uses, e.g. ``"PlantRidge"``. A
            Python identifier.
        cls: The regressor class.
        description: One line on what it is, for the docs.

    Returns:
        ``name``.

    Raises:
        SignalConfigError: If the name isn't an identifier, the class has no
            ``fit``, or another class is already registered under the name.
    """
    if not name.isidentifier():
        raise SignalConfigError(
            f"model name {name!r} must be a Python identifier, e.g. PlantRidge.",
            value=name,
        )
    if not callable(getattr(cls, "fit", None)):
        raise SignalConfigError(
            f"{cls!r} has no fit(X, y) method; register a regressor class that works "
            "like scikit-learn's.",
            value=name,
        )
    existing = _REGISTERED.get(name)
    if existing is not None and _qualified(existing[0]) != _qualified(cls):
        raise SignalConfigError(
            f"model {name!r} is already registered for {_qualified(existing[0])}; "
            "pick another name.",
            value=name,
        )
    _REGISTERED[name] = (cls, description)
    return name


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def unregister_model(name: str) -> None:
    """Remove a registered model if it is registered (for tests and notebooks)."""
    _REGISTERED.pop(name, None)


def registered_models() -> dict[str, str]:
    """Return the models registered with :func:`register_model`, with descriptions."""
    return {k: v[1] for k, v in sorted(_REGISTERED.items())}


def registered_model_class(name: str) -> type | None:
    """Return the class registered under ``name``, or None."""
    entry = _REGISTERED.get(name)
    return entry[0] if entry is not None else None


def is_linear_model(key: str) -> bool:
    """Return True if ``key`` is a built-in linear model or a registered one."""
    return key in LINEAR_MODELS or key in _REGISTERED


def check_model_key(key: str) -> str:
    """Return ``key`` if it names an accepted model, else raise ValueError."""
    if is_linear_model(key) or key == CURVE_FIT:
        return key
    known = ", ".join([*LINEAR_MODELS, CURVE_FIT, *_REGISTERED])
    raise ValueError(
        f"unknown model {key!r}. Use one of: {known}, or register your own with "
        "fs.register_model. Specs can't name other classes, because loading a spec "
        "must never import code."
    )


def fit_linear(
    key: str,
    X: np.ndarray,
    y: np.ndarray,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a linear model and return its coefficients.

    Args:
        key: One of :data:`LINEAR_MODELS`, or a model registered with
            :func:`~flexsignal.register_model`.
        X: Inputs, shape ``(n_rows, n_features)``, with no NaN.
        y: Targets, shape ``(n_rows,)``, with no NaN.
        params: Keyword arguments for the model's constructor, e.g.
            ``{"alpha": 1.0}`` for Ridge.

    Returns:
        ``{"intercept": float, "coef": [float, ...]}``.

    Raises:
        SignalConfigError: If ``params`` doesn't suit the model.
        SignalDataError: If there are fewer rows than coefficients to fit.
    """
    check_model_key(key)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < X.shape[1] + 1:
        raise SignalDataError(
            f"only {len(y)} rows to fit {X.shape[1] + 1} coefficients; widen the "
            "training data or relax the fit condition."
        )
    registered = _REGISTERED.get(key)
    cls = registered[0] if registered is not None else get_backend(key)
    params = dict(params or {})
    try:
        if registered is not None:
            model = cls(**params).fit(X, y)
            if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
                raise SignalConfigError(
                    f"model {key!r} has no coef_ and intercept_ after fit(); "
                    "register a regressor whose fitted coefficients describe it."
                )
            intercept = float(np.ravel(model.intercept_)[0])
            coef = np.ravel(model.coef_)
        elif key.startswith("sklearn:"):
            if key == "sklearn:TheilSenRegressor":
                params.setdefault("random_state", 0)
            model = cls(**params).fit(X, y)
            intercept, coef = float(model.intercept_), np.ravel(model.coef_)
        else:
            design = np.column_stack([np.ones(len(y)), X])
            # statsmodels only warns about arguments it doesn't know, and then
            # ignores them: a misspelled one is refused here instead.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = cls(y, design, **params)
            unknown = [
                str(w.message) for w in caught if "unknown kwargs" in str(w.message)
            ]
            if unknown:
                raise TypeError("; ".join(unknown))
            for w in caught:  # pass any other warnings on as they were
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            fitted = model.fit()
            intercept, coef = float(fitted.params[0]), np.asarray(fitted.params[1:])
    except TypeError as exc:
        raise SignalConfigError(
            f"model {key!r} doesn't accept params {sorted(params)}: {exc}.",
            field="params",
        ) from exc
    return {"intercept": intercept, "coef": [float(c) for c in coef]}


def predict_linear(state: Mapping[str, Any], X: np.ndarray) -> np.ndarray:
    """Apply coefficients from :func:`fit_linear`; rows with a NaN input give NaN."""
    X = np.asarray(X, dtype=float)
    coef = np.asarray(state["coef"], dtype=float)
    return float(state["intercept"]) + X @ coef


def fit_curve(
    formula: str,
    data: Any,
    y: np.ndarray,
    initial: Mapping[str, float],
    params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Fit a formula's parameters to ``y`` by least squares.

    Args:
        formula: e.g. ``"a + b * A"``. Every name that isn't a stream must be a
            key of ``initial``.
        data: The rows to fit on (a SignalFrame or DataFrame), with no NaN in
            the streams the formula uses.
        y: The values the formula should reproduce, one per row.
        initial: Starting value for each parameter, e.g. ``{"a": 0, "b": 1}``.
        params: Extra keyword arguments for :func:`scipy.optimize.curve_fit`, e.g.
            ``{"maxfev": 5000}``.

    Returns:
        The fitted parameters, ``{name: value}``.

    Raises:
        SignalConfigError: If ``params`` doesn't suit ``curve_fit`` (starting
            values go in ``initial``, not ``params={"p0": ...}``), or the formula
            gives no finite value at the starting values.
        SignalDataError: If there are too few rows, or the fit doesn't converge.
    """
    names: Sequence[str] = list(initial)
    params = dict(params or {})
    if "p0" in params:
        raise SignalConfigError(
            "don't pass p0 in params: the starting values come from initial, "
            "one per parameter, e.g. initial={'a': 0, 'b': 1}.",
            field="params",
            value=params["p0"],
        )
    if len(y) < len(names) + 1:
        raise SignalDataError(
            f"only {len(y)} rows to fit {len(names)} parameters; widen the training "
            "data or relax the fit condition."
        )
    curve_fit = get_backend(CURVE_FIT)
    p0 = [float(initial[n]) for n in names]

    def model(_x: Any, *p: float) -> np.ndarray:
        return evaluate_formula(data, formula, dict(zip(names, p, strict=True)))

    start = model(None, *p0)
    if not np.all(np.isfinite(start)):
        bad = int(np.count_nonzero(~np.isfinite(start)))
        raise SignalConfigError(
            f"formula {formula!r} gives NaN or infinity on {bad} of {len(start)} "
            f"rows at the initial values {dict(zip(names, p0, strict=True))}, so "
            "it can't be fitted. Pick initial values inside the formula's domain "
            "(e.g. log() of a positive number).",
            field="initial",
        )
    try:
        popt, _ = curve_fit(
            model,
            np.zeros(len(y)),
            np.asarray(y, dtype=float),
            p0=p0,
            **params,
        )
    except TypeError as exc:
        raise SignalConfigError(
            f"curve_fit doesn't accept params {sorted(params)}: {exc}.",
            field="params",
        ) from exc
    except RuntimeError as exc:
        raise SignalDataError(
            f"curve_fit didn't converge for {formula!r}: {exc}. Try better initial "
            "values, or params={'maxfev': 10000}."
        ) from exc
    return {n: float(v) for n, v in zip(names, popt, strict=True)}
