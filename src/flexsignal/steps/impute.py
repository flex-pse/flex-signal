"""Imputation steps: fill missing values and flag them as filled.

Every imputer fills only *gaps*, runs of missing values, and only those no
longer than its ``max_gap``. A gap's length is the time between the valid samples
either side of it. Values flagged ``EXCLUDED`` (deliberately left out) or
``NO_SAMPLE`` (added by a join) are never filled. Filled values keep the flags
that explain why they were missing, and gain ``IMPUTED_SHORT`` (interpolation)
or ``IMPUTED_MODEL`` (a model).

- ``interpolate``: between the valid values either side, for short gaps.
- ``kalman``: a structural state-space model's Kalman smoother (statsmodels).
- ``regression``: a linear model of other streams, their lags, and daily or
  weekly cycles.
- ``knn``: the k most similar rows (scikit-learn).
- ``iterative``: each stream modelled from the others in turn (scikit-learn).
- ``tiered``: route each gap to a method by its length.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from flexsignal.backends import get_backend
from flexsignal.backends.models import (
    check_model_key,
    fit_linear,
    is_linear_model,
    predict_linear,
)
from flexsignal.core.exceptions import SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import find_step, get_step_class, register_step
from flexsignal.spec.models import (
    Duration,
    SignedDuration,
    StrictModel,
    to_timedelta,
)
from flexsignal.steps._base import (
    StreamStep,
    check_streams,
    fourier_terms,
    gaps_to_fill,
    one_is_a_list,
    registered_lookup,
    regular_step,
    seconds,
    structural_args,
    whole_steps,
)

__all__ = [
    "KNN",
    "CycleTerm",
    "ImputeTier",
    "Interpolate",
    "Iterative",
    "Kalman",
    "Regression",
    "Tiered",
]

_METHOD_BACKENDS = {
    "linear": "numpy:interp",
    "nearest": "scipy:interp1d",
    "previous": "scipy:interp1d",
    "pchip": "scipy:PchipInterpolator",
    "akima": "scipy:Akima1DInterpolator",
}


def _max_gap_s(max_gap: str | None) -> float | None:
    return None if max_gap is None else to_timedelta(max_gap).total_seconds()


def _fill(
    sf: SignalFrame, filled: Mapping[str, tuple[np.ndarray, np.ndarray]], flag: Flag
) -> SignalFrame:
    """Write ``{stream: (where, new_values)}`` into the frame and add ``flag``."""
    values, flags = sf.values, sf.flags
    for col, (where, new) in filled.items():
        where = where & ~np.isnan(new)
        if not where.any():
            continue
        arr = values[col].to_numpy(copy=True)
        col_flags = flags[col].to_numpy(copy=True)
        arr[where] = new[where]
        col_flags[where] |= FLAG_DTYPE(int(flag))
        values[col] = arr
        flags[col] = col_flags
    return sf.evolve(values=values, flags=flags)


def _standardize(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return (block - mean) / std, mean, std


class _GapImputer(StreamStep):
    """Shared options: which gaps to fill."""

    max_gap: Duration | None = Field(
        None,
        description=(
            "Longest gap to fill, measured between the valid samples either side, e.g. "
            "6h. Default: every gap, including those at the start and end."
        ),
    )

    def _targets(self, sf: SignalFrame) -> dict[str, np.ndarray]:
        check_streams(sf, self.streams)
        t = seconds(sf.index)
        limit = _max_gap_s(self.max_gap)
        values, flags = sf.values, sf.flags
        return {
            col: gaps_to_fill(values[col].to_numpy(), flags[col].to_numpy(), t, limit)
            for col in self.streams
        }


# -- interpolate ------------------------------------------------------------------


def _interpolator(method: str, x: np.ndarray, y: np.ndarray):
    backend = get_backend(_METHOD_BACKENDS[method])
    if method == "linear":
        return lambda xi: backend(xi, x, y)
    if method in ("nearest", "previous"):
        return backend(x, y, kind=method, assume_sorted=True)
    return backend(x, y)


@register_step
class Interpolate(_GapImputer):
    """Fill short interior gaps by interpolating between the valid values either side.

    Only gaps no longer than ``max_gap`` are filled: on 5-minute data,
    ``max_gap="15min"`` fills up to two missing samples in a row. Gaps at the
    start or end of the data are never filled. Filled values gain
    ``IMPUTED_SHORT``.

    Methods use numpy and scipy:

    - ``linear``: :func:`numpy.interp`;
    - ``nearest`` and ``previous``: :class:`scipy.interpolate.interp1d`;
    - ``pchip``: :class:`scipy.interpolate.PchipInterpolator`, shape-preserving
      with no overshoot;
    - ``akima``: :class:`scipy.interpolate.Akima1DInterpolator`.
    """

    kind: Literal["interpolate"] = "interpolate"
    max_gap: Duration = Field(  # type: ignore[assignment]
        description=(
            "Longest gap to fill, measured between the valid samples either side, e.g. "
            "15min."
        )
    )
    method: Literal["linear", "nearest", "previous", "pchip", "akima"] = Field(
        "linear", description="Interpolation method."
    )

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill short gaps (see the class docstring)."""
        targets = self._targets(sf)
        t = seconds(sf.index)
        values = sf.values
        filled = {}
        for col, where in targets.items():
            arr = values[col].to_numpy()
            valid = ~np.isnan(arr)
            if valid.sum() < 2 or not where.any():
                continue
            interp = _interpolator(self.method, t[valid], arr[valid])
            new = np.full(len(arr), np.nan)
            new[where] = np.asarray(interp(t[where]), dtype=float)
            filled[col] = (where, new)
        return _fill(sf, filled, Flag.IMPUTED_SHORT)


# -- kalman -------------------------------------------------------------------------


class CycleTerm(StrictModel):
    """A repeating cycle, such as a day or a week, as a few sine and cosine waves.

    Used by ``kalman`` and ``structural`` (``seasonal``) and by both
    ``regression`` steps (``fourier``).
    """

    period: Duration = Field(description="Length of one cycle, e.g. 1D or 7D.")
    harmonics: int | None = Field(
        None,
        ge=1,
        le=20,
        description=(
            "How many sine/cosine pairs describe its shape. Default: 2 for kalman "
            "and structural, 3 for regression."
        ),
    )

    def pairs(self, default: int) -> int:
        """Return ``harmonics``, or the using step's ``default`` when it is unset."""
        return self.harmonics if self.harmonics is not None else default


@register_step
class Kalman(_GapImputer):
    """Fill gaps with the Kalman smoother of a structural time-series model.

    Each stream is modelled as a level (``local_level``), or a level with a slope
    (``local_linear_trend``), plus optional ``seasonal`` cycles and noise, using
    :class:`statsmodels.tsa.statespace.structural.UnobservedComponents`. ``fit``
    estimates the model's variances on the training data by maximum likelihood
    and stores them as JSON. ``transform`` runs the Kalman smoother with those
    values, which estimates every missing value from the data on both sides of
    it. Filled values gain ``IMPUTED_MODEL``.

    It needs a regular time index; run ``align`` or ``resample`` first. A long
    gap relaxes towards the level and cycles, so set ``seasonal`` when the
    stream has a daily pattern.
    """

    kind: Literal["kalman"] = "kalman"
    level: Literal["local_level", "local_linear_trend"] = Field(
        "local_level", description="How the underlying level moves."
    )
    seasonal: tuple[CycleTerm, ...] = Field(
        (), description="Repeating cycles, e.g. [{period: 1D, harmonics: 2}]."
    )

    def _model(self, series: np.ndarray, step: pd.Timedelta):
        cls = get_backend("statsmodels:UnobservedComponents")
        return cls(series, **structural_args(self.level, self.seasonal, step))

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Estimate each stream's model variances on the training data."""
        check_streams(sf, self.streams)
        step = regular_step(sf.index, kind="kalman")
        params = {}
        for col in self.streams:
            series = sf.values[col].to_numpy()
            if np.count_nonzero(~np.isnan(series)) < 10:
                raise SignalDataError(
                    f"kalman: stream {col!r} has fewer than 10 values to fit on."
                )
            model = self._model(series, step)
            result = model.fit(disp=False)
            params[col] = {
                name: float(value)
                for name, value in zip(model.param_names, result.params, strict=True)
            }
        return {"params": params}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill gaps with the smoothed signal (see the class docstring)."""
        targets = self._targets(sf)
        step = regular_step(sf.index, kind="kalman")
        values = sf.values
        filled = {}
        for col, where in targets.items():
            if not where.any():
                continue
            model = self._model(values[col].to_numpy(), step)
            fitted = state["params"][col]
            result = model.smooth(np.array([fitted[n] for n in model.param_names]))
            smoothed = np.asarray(result.smoother_results.smoothed_forecasts)[0]
            filled[col] = (where, smoothed)
        return _fill(sf, filled, Flag.IMPUTED_MODEL)


# -- regression ---------------------------------------------------------------------


@register_step
class Regression(_GapImputer):
    """Fill gaps from a linear model of other streams and of time of day or week.

    The inputs are the ``features`` streams, each also at every lag in ``lags``,
    and sine and cosine terms for each ``fourier`` cycle. ``fit`` learns one model
    per stream on the training rows where the stream and all its inputs have
    values, and stores the coefficients as JSON. ``transform`` fills a gap
    wherever the inputs have values. Filled values gain ``IMPUTED_MODEL``.

    Models are the linear models of
    :data:`~flexsignal.backends.models.LINEAR_MODELS`, e.g. ``sklearn:Ridge``.
    """

    kind: Literal["regression"] = "regression"
    features: tuple[str, ...] = Field(
        (), description="Other streams the model uses, e.g. [pressure]."
    )
    lags: tuple[SignedDuration, ...] = Field(
        (),
        description=(
            "Also use each feature this long before, e.g. [5min, 15min]. Needs a "
            "regular index."
        ),
    )
    fourier: tuple[CycleTerm, ...] = Field(
        (), description="Cycles to model, e.g. [{period: 1D, harmonics: 3}]."
    )
    model: str = Field(
        "sklearn:Ridge", description="A linear model key, e.g. sklearn:Ridge."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the model."
    )

    @field_validator("model")
    @classmethod
    def _linear_model(cls, value: str) -> str:
        check_model_key(value)
        if not is_linear_model(value):
            raise ValueError(f"regression needs a linear model, not {value!r}")
        return value

    _one_is_a_list = field_validator("features", "lags", mode="before")(one_is_a_list)

    @model_validator(mode="after")
    def _has_inputs(self) -> Regression:
        if not self.features and not self.fourier:
            raise ValueError("give features, fourier terms, or both")
        if self.lags and not self.features:
            raise ValueError("lags apply to features, so give features too")
        overlap = sorted(set(self.features) & set(self.streams))
        if overlap:
            raise ValueError(
                f"{overlap} can't be both filled and used as a feature; to use a "
                "stream's own history, use kalman"
            )
        return self

    def _design(self, sf: SignalFrame) -> np.ndarray:
        check_streams(sf, self.features, field="features")
        values = sf.values
        columns = [values[f].to_numpy() for f in self.features]
        if self.lags:
            step = regular_step(sf.index, kind="regression with lags")
            for lag in self.lags:
                periods = whole_steps(
                    to_timedelta(lag), step, what=f"lag {lag}", field="lags"
                )
                for f in self.features:
                    columns.append(values[f].shift(periods).to_numpy())
        terms = [(to_timedelta(t.period), t.pairs(3)) for t in self.fourier]
        cycles = fourier_terms(sf.index, terms)
        parts = [np.column_stack(columns)] if columns else []
        return np.column_stack([*parts, cycles]) if parts else cycles

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Fit one model per stream on the training data."""
        check_streams(sf, self.streams)
        X = self._design(sf)
        usable_X = ~np.isnan(X).any(axis=1)
        models = {}
        for col in self.streams:
            y = sf.values[col].to_numpy()
            rows = usable_X & ~np.isnan(y)
            models[col] = fit_linear(self.model, X[rows], y[rows], self.params)
        return {"models": models}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill gaps from the fitted models (see the class docstring)."""
        targets = self._targets(sf)
        X = self._design(sf)
        filled = {
            col: (where, predict_linear(state["models"][col], X))
            for col, where in targets.items()
            if where.any()
        }
        return _fill(sf, filled, Flag.IMPUTED_MODEL)


# -- knn and iterative --------------------------------------------------------------


class _MatrixImputer(_GapImputer):
    """Imputers that fill a table of streams, fitted on the data they fill."""

    features: tuple[str, ...] = Field(
        (),
        description=(
            "Other streams that help find similar rows; they are used, not filled."
        ),
    )

    _one_feature_is_a_list = field_validator("features", mode="before")(one_is_a_list)

    def _impute(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill gaps (see the class docstring)."""
        targets = self._targets(sf)
        check_streams(sf, self.features, field="features")
        if not any(w.any() for w in targets.values()):
            return sf
        columns = list(dict.fromkeys([*self.streams, *self.features]))
        block = sf.values[columns].to_numpy(dtype="float64")
        keep = ~np.isnan(block).all(axis=0)
        if not keep.any():
            return sf  # every stream is empty: nothing to learn from
        scaled, mean, std = _standardize(block[:, keep])
        result = np.full(block.shape, np.nan)
        result[:, keep] = self._impute(scaled) * std + mean
        filled = {
            col: (where, result[:, columns.index(col)])
            for col, where in targets.items()
        }
        return _fill(sf, filled, Flag.IMPUTED_MODEL)


@register_step
class KNN(_MatrixImputer):
    """Fill a value from the rows whose other streams are most similar.

    Uses :class:`sklearn.impute.KNNImputer` on the ``streams`` and ``features``,
    each scaled to unit variance first so units don't decide the distance. It
    learns from the data it fills, so ``fit`` stores nothing. Filled values gain
    ``IMPUTED_MODEL``. A row needs at least one other stream with a value.
    """

    kind: Literal["knn"] = "knn"
    n_neighbors: int = Field(5, ge=1, description="How many similar rows to use.")
    weights: Literal["uniform", "distance"] = Field(
        "uniform", description="uniform: plain mean; distance: closer rows count more."
    )

    def _impute(self, block: np.ndarray) -> np.ndarray:
        imputer = get_backend("sklearn:KNNImputer")(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            keep_empty_features=True,
        )
        return imputer.fit_transform(block)


@register_step
class Iterative(_MatrixImputer):
    """Fill each stream from a model of the others, round-robin (MICE-style).

    Uses :class:`sklearn.impute.IterativeImputer` with its default Bayesian
    ridge model, on the ``streams`` and ``features`` scaled to unit variance. It
    learns from the data it fills, so ``fit`` stores nothing; the random state is
    fixed, so results are repeatable. Filled values gain ``IMPUTED_MODEL``.
    """

    kind: Literal["iterative"] = "iterative"
    max_iter: int = Field(10, ge=1, le=100, description="Rounds of modelling.")

    def _impute(self, block: np.ndarray) -> np.ndarray:
        imputer = get_backend("sklearn:IterativeImputer")(
            max_iter=self.max_iter, random_state=0, keep_empty_features=True
        )
        return imputer.fit_transform(block)


# -- tiered -------------------------------------------------------------------------

_TIER_METHODS = ("interpolate", "kalman", "regression", "knn", "iterative")


class ImputeTier(StrictModel):
    """One tier of a ``tiered`` imputer: a method for gaps up to a length."""

    max_gap: Duration | None = Field(
        description=(
            "Longest gap this tier fills, e.g. 15min. null on the last tier means "
            "every gap left."
        )
    )
    method: str = Field(
        description=(
            "interpolate, kalman, regression, knn, iterative, or the name of a "
            "registered imputer."
        )
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="The method's other options, e.g. {method: pchip}.",
    )

    @field_validator("method")
    @classmethod
    def _known_method(cls, value: str) -> str:
        if value in _TIER_METHODS:
            return value
        step = find_step(value)
        method = getattr(step, "registered", None)
        if getattr(method, "role", None) == "imputer":
            return value
        raise ValueError(
            f"unknown tier method {value!r}; use one of {list(_TIER_METHODS)} or the "
            "name of a registered imputer (fs.register_imputer)"
        )


@register_step
class Tiered(StreamStep):
    """Route each gap to an imputer by its length.

    Short gaps are best interpolated; long ones need a model. For example::

        tiers=[
            {"max_gap": "15min", "method": "interpolate"},
            {"max_gap": "6h", "method": "regression",
             "options": {"features": ["pressure"]}},
            {"max_gap": None, "method": "kalman"},
        ]

    A gap goes to the first tier whose ``max_gap`` covers its length, measured in
    the data as it arrives at this step. Each tier's imputer is fitted on the
    training data, and its state is kept under ``tiers``. Filled values carry the
    flag of the method that filled them.
    """

    kind: Literal["tiered"] = "tiered"
    tiers: tuple[ImputeTier, ...] = Field(
        min_length=1, description="The tiers, shortest max_gap first."
    )

    @model_validator(mode="after")
    def _ordered_tiers(self) -> Tiered:
        limits = [t.max_gap for t in self.tiers]
        if any(limit is None for limit in limits[:-1]):
            raise ValueError("only the last tier may have max_gap null")
        bounded = [to_timedelta(limit) for limit in limits if limit is not None]
        if bounded != sorted(bounded) or len(set(bounded)) != len(bounded):
            raise ValueError("tiers must be ordered by increasing max_gap")
        for i in range(len(self.tiers)):
            self._inner(i)  # each tier's options must make a valid step
        return self

    def _inner(self, i: int) -> StreamStep:
        tier = self.tiers[i]
        if "streams" in tier.options or "max_gap" in tier.options:
            raise ValueError(
                f"tier {i}: streams and max_gap are set by the tiered step, not "
                "in options"
            )
        cls = get_step_class(tier.method)
        try:
            return cls(streams=self.streams, max_gap=tier.max_gap, **tier.options)
        except Exception as exc:
            raise ValueError(f"tier {i} ({tier.method}): {exc}") from None

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Fit each tier's imputer on the training data."""
        check_streams(sf, self.streams)
        return {"tiers": [self._inner(i).fit(sf) for i in range(len(self.tiers))]}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill each gap with its tier's method (see the class docstring)."""
        check_streams(sf, self.streams)
        values, flags = sf.values, sf.flags
        for i in range(len(self.tiers)):
            out = self._inner(i).transform(sf, state["tiers"][i])
            new_values, new_flags = out.values, out.flags
            for col in self.streams:
                current = values[col].to_numpy(copy=True)
                proposed = new_values[col].to_numpy()
                take = np.isnan(current) & ~np.isnan(proposed)
                if not take.any():
                    continue
                current[take] = proposed[take]
                col_flags = flags[col].to_numpy(copy=True)
                col_flags[take] = new_flags[col].to_numpy()[take]
                values[col] = current
                flags[col] = col_flags
        return sf.evolve(values=values, flags=flags)


__getattr__ = registered_lookup(__name__, "imputer")
