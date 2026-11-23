"""Correction steps: fix a stream's systematic error using a trusted reference.

``calibrate`` solves the A/B/C/D case. Stream A always reports, but with an
error that varies. Stream B measures the same quantity accurately, but only some
of the time, and only while some condition holds. ``calibrate`` learns the
mapping from A to B on the rows where B can be trusted, then applies it to A
everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from flexsignal.backends.models import (
    CURVE_FIT,
    check_model_key,
    fit_curve,
    fit_linear,
    predict_linear,
)
from flexsignal.core.conditions import (
    Condition,
    Formula,
    evaluate_condition,
    evaluate_formula,
    formula_names,
)
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import register_step
from flexsignal.spec.models import Duration, StepSpec, StrictModel, to_timedelta
from flexsignal.steps._base import check_streams, one_is_a_list

__all__ = ["Calibrate", "RollingRefit"]


class RollingRefit(StrictModel):
    """Refit the calibration on a moving window, so it follows drift."""

    rolling: Duration = Field(
        description=(
            "Width of the window of trusted reference rows each refit uses, e.g. 1D."
        )
    )
    every: Duration = Field(
        "1D",
        description=(
            "How often to refit. Each block of this length uses a model fitted on "
            "the window that ends where the block ends."
        ),
    )
    min_rows: int = Field(
        10,
        ge=2,
        description=(
            "Fewest trusted rows a window needs. Blocks with fewer use the model "
            "fitted on the training data."
        ),
    )


@register_step
class Calibrate(StepSpec):
    """Correct a stream against an intermittent, trusted reference.

    1. **Fit.** On the *trusted rows*, the rows where the reference has a value
       and ``fit_when`` holds, fit a model that maps the ``features`` (by default
       the target itself) to the reference.
    2. **Apply.** Predict the corrected value wherever ``apply_when`` holds and
       the features have values. These values are flagged ``CORRECTED``, on top
       of the target's own flags. Where ``apply_when`` is false, the target's
       value passes through unchanged.
    3. **Prefer the reference.** With ``prefer_reference``, the trusted rows take
       the reference's own value and flags instead of a prediction.
    4. **Follow drift.** With ``refit={"rolling": "1D"}``, the model is refitted
       for each block of time on the trusted rows of the window ending with that
       block. The model fitted on the training data covers blocks whose window
       has too few trusted rows.

    The fitted state holds the coefficients (or the formula's parameters) and fit
    metrics: rows used, RMSE and R².

    Models (``model``): the linear models of
    :data:`~flexsignal.backends.models.LINEAR_MODELS`, e.g.
    ``sklearn:HuberRegressor``, or ``scipy:curve_fit`` with a ``formula`` such as
    ``"a + b * A + c * A ** 2"`` and ``initial`` values for its parameters.

    With ``mode="residual"`` the model predicts the error, ``reference - target``,
    and the corrected value is ``target + prediction``. Use it when the features
    explain the error rather than the value, e.g. a temperature that biases A.
    """

    kind: Literal["calibrate"] = "calibrate"
    target: str = Field(description="The stream to correct, e.g. A.")
    reference: str = Field(
        description="The trusted stream measuring the same quantity, e.g. B."
    )
    features: tuple[str, ...] | None = Field(
        None,
        description=(
            "Model inputs (default: the target). Not used with scipy:curve_fit, "
            "whose formula names its inputs."
        ),
    )
    output: str | None = Field(
        None,
        description=(
            "Name of the corrected stream (default: <target>_clean). Set it to the "
            "target's name to replace the target."
        ),
    )
    fit_when: Condition | None = Field(
        None,
        description=(
            "When the reference can be trusted, e.g. 'C == 0 and D == 0 and B > 0'. "
            "Default: wherever it has a value."
        ),
    )
    apply_when: Condition | None = Field(
        None,
        description=(
            "Where to correct the target (default: everywhere). Elsewhere the "
            "target's value passes through unchanged."
        ),
    )
    model: str = Field(
        "sklearn:LinearRegression",
        description=(
            "The model: a linear model key such as sklearn:HuberRegressor or "
            "statsmodels:RLM, or scipy:curve_fit with a formula."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for the model, e.g. {alpha: 1.0} for Ridge.",
    )
    formula: Formula | None = Field(
        None,
        description=(
            "For scipy:curve_fit: the model as a formula of streams and parameters, "
            "e.g. 'a + b * A'."
        ),
    )
    initial: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "For scipy:curve_fit: a starting value for each parameter in the "
            "formula, e.g. {a: 0, b: 1}."
        ),
    )
    mode: Literal["map", "residual"] = Field(
        "map",
        description=(
            "map: predict the reference value; residual: predict reference - target "
            "and add it to the target."
        ),
    )
    prefer_reference: bool = Field(
        True,
        description="Use the reference's own value on the rows where it is trusted.",
    )
    refit: Literal["static"] | RollingRefit = Field(
        "static",
        description=(
            "static: one model fitted on the training data; or {rolling: 1D} to "
            "refit on a moving window and follow drift."
        ),
    )
    units: str | None = Field(
        None, description="Units of the output (default: the target's units)."
    )

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        return check_model_key(value)

    _one_feature_is_a_list = field_validator("features", mode="before")(one_is_a_list)

    @model_validator(mode="after")
    def _formula_with_curve_fit(self) -> Calibrate:
        if self.model == CURVE_FIT:
            if self.formula is None or not self.initial:
                raise ValueError(
                    "scipy:curve_fit needs a formula and initial values for its "
                    "parameters, e.g. formula='a + b * A', initial={a: 0, b: 1}"
                )
            if self.features is not None:
                raise ValueError(
                    "with scipy:curve_fit the formula names the inputs; drop features"
                )
            # The formula must be valid with these parameter names.
            formula_names(self.formula, self.initial)
        elif self.formula is not None or self.initial:
            raise ValueError("formula and initial are only used with scipy:curve_fit")
        if self.reference == self.target:
            raise ValueError("target and reference must be different streams")
        return self

    # -- helpers ---------------------------------------------------------------------

    @property
    def output_name(self) -> str:
        """The name of the corrected stream."""
        return self.output or f"{self.target}_clean"

    def _inputs(self) -> list[str]:
        if self.model == CURVE_FIT:
            return formula_names(self.formula or "", self.initial)
        return list(self.features or (self.target,))

    def _check(self, sf: SignalFrame) -> None:
        check_streams(sf, [self.target], field="target")
        check_streams(sf, [self.reference], field="reference")
        check_streams(sf, self._inputs(), field="features")
        if self.output_name != self.target and self.output_name in sf:
            raise SignalConfigError(
                f"stream {self.output_name!r} already exists; pick another output, or "
                "set output to the target's name to replace it.",
                field="output",
            )

    def _trusted(self, sf: SignalFrame) -> np.ndarray:
        trusted = sf.values[self.reference].notna().to_numpy()
        if self.fit_when is not None:
            trusted = (
                trusted
                & evaluate_condition(sf, self.fit_when, field="fit_when").to_numpy()
            )
        return trusted

    def _fit_rows(self, sf: SignalFrame, rows: np.ndarray) -> dict[str, Any]:
        values = sf.values.loc[rows]
        inputs = self._inputs()
        # The target is needed only where it is an input: in residual mode, or
        # when the features (or the formula) name it.
        needed = inputs + [self.target] if self.mode == "residual" else inputs
        usable = values[needed].notna().all(axis=1).to_numpy()
        values = values.loc[usable]
        y = values[self.reference].to_numpy()
        if self.mode == "residual":
            y = y - values[self.target].to_numpy()
        if self.model == CURVE_FIT:
            fitted = fit_curve(self.formula or "", values, y, self.initial, self.params)
            predicted = evaluate_formula(values, self.formula or "", fitted)
            model: dict[str, Any] = {"parameters": fitted}
        else:
            X = values[inputs].to_numpy()
            model = fit_linear(self.model, X, y, self.params)
            predicted = predict_linear(model, X)
        residual = y - predicted
        spread = float(np.sum((y - y.mean()) ** 2))
        model["metrics"] = {
            "rows": int(len(y)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - np.sum(residual**2) / spread) if spread else None,
        }
        return model

    def _predict(self, values: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
        if self.model == CURVE_FIT:
            pred = evaluate_formula(values, self.formula or "", model["parameters"])
        else:
            pred = predict_linear(model, values[self._inputs()].to_numpy())
        if self.mode == "residual":
            pred = values[self.target].to_numpy() + pred
        return pred

    # -- the step ---------------------------------------------------------------------

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Fit the model on the training data's trusted rows."""
        self._check(sf)
        trusted = self._trusted(sf)
        if not trusted.any():
            raise SignalDataError(
                f"no trusted rows to fit on: {self.reference!r} never has a value "
                f"where fit_when ({self.fit_when!r}) holds."
            )
        state = self._fit_rows(sf, trusted)
        metrics = state.pop("metrics")
        return {"model": state, "metrics": metrics}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Write the corrected stream (see the class docstring)."""
        self._check(sf)
        trusted = self._trusted(sf)
        if isinstance(self.refit, RollingRefit):
            pred = self._rolling_predict(sf, trusted, state["model"])
        else:
            pred = self._predict(sf.values, state["model"])

        values, flags = sf.values, sf.flags
        target_v = values[self.target].to_numpy()
        target_f = flags[self.target].to_numpy().astype("int64")
        out_v = np.full(len(sf), np.nan)
        out_f = np.full(len(sf), int(Flag.MISSING), dtype="int64")

        apply = np.ones(len(sf), dtype=bool)
        if self.apply_when is not None:
            apply = evaluate_condition(
                sf, self.apply_when, field="apply_when"
            ).to_numpy()
            out_v[~apply] = target_v[~apply]
            out_f[~apply] = target_f[~apply]
        has_pred = apply & ~np.isnan(pred)
        out_v[has_pred] = pred[has_pred]
        out_f[has_pred] = target_f[has_pred] | int(Flag.CORRECTED)
        no_pred = apply & np.isnan(pred)
        out_f[no_pred] = target_f[no_pred] | int(Flag.MISSING)
        if self.prefer_reference:
            out_v[trusted] = values[self.reference].to_numpy()[trusted]
            out_f[trusted] = flags[self.reference].to_numpy()[trusted]

        name = self.output_name
        values[name] = out_v
        flags[name] = out_f.astype(FLAG_DTYPE)
        units = self.units if self.units is not None else sf.meta[self.target].units
        meta = dict(sf.meta)
        base = meta.get(name) or meta[self.target]
        meta[name] = base.model_copy(update={"units": units})
        return sf.evolve(values=values, flags=flags, meta=meta)

    def _rolling_predict(
        self, sf: SignalFrame, trusted: np.ndarray, fallback: Mapping[str, Any]
    ) -> np.ndarray:
        assert isinstance(self.refit, RollingRefit)
        every = to_timedelta(self.refit.every)
        window = to_timedelta(self.refit.rolling)
        index = sf.index
        values = sf.values
        block = index.floor(every)
        pred = np.full(len(sf), np.nan)
        for start in block.unique():
            in_block = np.asarray(block == start)
            end = start + every
            rows = trusted & (index >= end - window) & (index < end)
            model: Mapping[str, Any] = fallback
            if rows.sum() >= self.refit.min_rows:
                try:
                    model = self._fit_rows(sf, rows)
                except SignalDataError:
                    model = fallback
            pred[in_block] = self._predict(values.loc[in_block], model)
        return pred
