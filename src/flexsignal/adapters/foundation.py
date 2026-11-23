"""Zero-shot forecasting with pretrained foundation models (the ``[foundation]`` extra).

``fs.forecast.ChronosForecast`` forecasts a stream with a pretrained model and no
training: the model reads the recent history (and the ``exog`` streams, as
covariates) and returns a forecast with quantiles. The default is Chronos-2
(Apache-2.0). Nothing is learned or stored: the model is named by its Hugging
Face id and a pinned revision (a commit), so a spec gives the same forecast
wherever it runs. The weights are downloaded once into the Hugging Face cache.

TimesFM 2.5 (Apache-2.0) can be named too; TimesFM itself is an optional add-on
(``pip install timesfm``).

``fs.forecast.T0Forecast`` does the same with The Forecasting Company's t0-alpha
(Apache-2.0), from the ``[t0]`` extra. ``tfc-t0`` needs ``huggingface-hub>=1``
and PyPOTS 2.0 needs ``huggingface-hub<1``, so ``[t0]`` can't be installed with
``[pypots]``, and ``[all]`` leaves it out.

Every model here has open (Apache-2.0) weights. flex-signal doesn't ship
adapters to models whose licences restrict commercial use: register one
yourself (:func:`flexsignal.register_forecaster`) if its licence allows your
use.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    quantile_key,
    regular_step,
    require,
    seeded,
    steps_in,
)
from flexsignal.backends import register_backend
from flexsignal.core.exceptions import (
    SignalBackendError,
    SignalConfigError,
    SignalDataError,
)
from flexsignal.steps.registered import register_forecaster

__all__ = ["PINNED_REVISIONS", "chronos_forecast", "t0_forecast"]

EXTRA = "foundation"
METHOD = "ChronosForecast"
T0_EXTRA = "t0"
T0_METHOD = "T0Forecast"

ModelId = Literal[
    "amazon/chronos-2",
    "autogluon/chronos-2-small",
    "google/timesfm-2.5-200m-pytorch",
]
T0ModelId = Literal["theforecastingcompany/t0-alpha"]

PINNED_REVISIONS: dict[str, str] = {
    "amazon/chronos-2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "autogluon/chronos-2-small": "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
    "google/timesfm-2.5-200m-pytorch": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
    "theforecastingcompany/t0-alpha": "741927847fe02367624749bb9290b2f09f128e7d",
}
"""The Hugging Face commit each model runs at when ``revision`` is not given."""

# Every whole-percent quantile: flex-signal picks the ones a step asks for.
_LEVELS = [round(i / 100, 2) for i in range(1, 100)]


def _grid(
    history: pd.DataFrame, future: pd.DataFrame, method: str = METHOD
) -> tuple[pd.Timedelta, int]:
    """Return the data's step and how many steps ahead the last forecast row is."""
    step = regular_step(history.index, method)
    ahead = (future.index - history.index[-1]) / step
    if len(future) == 0 or not np.allclose(ahead, np.arange(1, len(future) + 1)):
        raise SignalDataError(
            f"{method}: the forecast rows must follow the history one data step "
            f"({step}) apart."
        )
    return step, len(future)


def _context(
    history: pd.DataFrame,
    step: pd.Timedelta,
    context: str | None,
    method: str = METHOD,
) -> pd.DataFrame:
    """The history the model reads: the last ``context`` of it, from its first value."""
    if context is not None:
        n = steps_in(context, step, field="context", method=method)
        history = history.iloc[-n:]
    target = history.iloc[:, 0]
    if target.notna().sum() < 2:
        where = "its history" if context is None else f"the last {context} (context)"
        raise SignalDataError(
            f"{method}: {target.name!r} needs at least two values in {where}; "
            + ("give it more history." if context is None else "use a longer context.")
        )
    return history.loc[target.first_valid_index() :]


def _revision(model: str, revision: str | None, method: str) -> str:
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise SignalConfigError(
            f"{method}: revision must be a Hugging Face commit or tag, e.g. "
            f"{PINNED_REVISIONS[model]!r}.",
            field="revision",
            value=revision,
        )
    return revision or PINNED_REVISIONS[model]


@lru_cache(maxsize=2)
def _chronos(model: str, revision: str) -> Any:
    chronos = require("chronos", EXTRA, method=METHOD)
    require("torch", EXTRA, method=METHOD)
    try:
        return chronos.Chronos2Pipeline.from_pretrained(
            model, revision=revision, device_map="cpu"
        )
    except (OSError, ValueError) as exc:
        raise SignalBackendError(
            f"{METHOD}: could not load {model} at revision {revision} from Hugging "
            f"Face: {exc}. Check the network connection, or the revision.",
            value=model,
        ) from exc


def _run_chronos(
    history: pd.DataFrame, future: pd.DataFrame, model: str, revision: str, steps: int
) -> dict[str, np.ndarray]:
    # Load first: a missing chronos or torch raises the error naming the extra.
    pipeline = _chronos(model, revision)
    import torch

    target = history.iloc[:, 0].to_numpy(dtype="float64")
    inputs: dict[str, Any] = {"target": target}
    if history.shape[1] > 1:
        inputs["past_covariates"] = {
            str(c): history[c].to_numpy(dtype="float64") for c in history.columns[1:]
        }
        inputs["future_covariates"] = {
            str(c): future[c].to_numpy(dtype="float64") for c in history.columns[1:]
        }
    with seeded(0), torch.inference_mode():
        quantiles, _ = pipeline.predict_quantiles(
            [inputs], prediction_length=steps, quantile_levels=_LEVELS
        )
    q = quantiles[0][0].to(torch.float64).numpy()  # (steps, levels)
    out = {quantile_key(level): q[:, i] for i, level in enumerate(_LEVELS)}
    out["mean"] = out["q50"]
    return out


@lru_cache(maxsize=1)
def _timesfm(model: str, revision: str) -> Any:
    try:
        timesfm = importlib.import_module("timesfm")
    except ImportError as exc:
        raise SignalBackendError(
            f"{METHOD}: {model} needs 'timesfm', which is not installed: install it "
            "with `pip install timesfm` (an optional add-on of "
            "flex-signal[foundation]).",
            value=model,
        ) from exc
    return timesfm, timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model, revision=revision
    )


def _run_timesfm(
    history: pd.DataFrame, model: str, revision: str, steps: int
) -> dict[str, np.ndarray]:
    if history.shape[1] > 1:
        raise SignalConfigError(
            f"{METHOD}: TimesFM models don't take exog streams here; drop exog or "
            "use model='amazon/chronos-2', which uses them as covariates.",
            field="exog",
        )
    timesfm, tfm = _timesfm(model, revision)
    tfm.compile(timesfm.ForecastConfig(max_context=len(history), max_horizon=steps))
    y = history.iloc[:, 0].interpolate(limit_direction="both").to_numpy("float64")
    with seeded(0):
        point, quantiles = tfm.forecast(horizon=steps, inputs=[y])
    q = np.asarray(quantiles, dtype="float64")[0]  # (steps, mean + q10..q90)
    out = {quantile_key(i / 10): q[:, i] for i in range(1, 10)}
    out["mean"] = np.asarray(point, dtype="float64")[0]
    return out


def chronos_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: ModelId = "amazon/chronos-2",
    revision: str | None = None,
    context: str | None = None,
) -> dict[str, np.ndarray]:
    """Forecast with a pretrained foundation model, zero-shot (no training).

    The model reads the stream's recent history, and the ``exog`` streams as
    covariates (their history and their known future values), and forecasts the
    horizon with quantiles from its own predictive distribution; the forecast's
    value is the median. Use it for a good forecast without choosing or tuning a
    model, when there's too little history to train one, or as a strong
    baseline. It runs on the CPU and gives the same result every run. The
    weights are downloaded from Hugging Face the first time (about 0.5 GB for
    Chronos-2) and cached.

    Args:
        history: The stream (first column) and the exog streams, up to the
            issue time; given by flex-signal.
        future: The exog streams over the forecast rows; given by flex-signal.
        model: The pretrained model, by Hugging Face id: ``amazon/chronos-2``
            (default, Apache-2.0), ``autogluon/chronos-2-small`` (smaller and
            faster, Apache-2.0) or ``google/timesfm-2.5-200m-pytorch``
            (Apache-2.0, needs ``pip install timesfm``, no exog).
        revision: The model's Hugging Face commit. Default: the commit this
            version of flex-signal pins for the model, so results don't change
            when the model is updated upstream.
        context: How much history the model reads, as a duration (e.g. ``14D``).
            Default: all of it, up to the model's limit (8192 steps for
            Chronos-2).

    Returns:
        ``{"mean": ..., "q01": ..., ..., "q99": ...}``: the median and every
        whole-percent quantile over the forecast rows.
    """
    revision = _revision(model, revision, METHOD)
    step, steps = _grid(history, future)
    window = _context(history, step, context)
    if model.startswith("google/timesfm"):
        return _run_timesfm(window, model, revision, steps)
    return _run_chronos(window, future, model, revision, steps)


@lru_cache(maxsize=1)
def _t0(model: str, revision: str) -> Any:
    t0 = require("t0", T0_EXTRA, method=T0_METHOD)
    require("torch", T0_EXTRA, method=T0_METHOD)
    try:
        return t0.T0Forecaster.from_pretrained(model, revision=revision).eval()
    except (OSError, ValueError) as exc:
        raise SignalBackendError(
            f"{T0_METHOD}: could not load {model} at revision {revision} from "
            f"Hugging Face: {exc}. Check the network connection, or the revision.",
            value=model,
        ) from exc


def t0_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: T0ModelId = "theforecastingcompany/t0-alpha",
    revision: str | None = None,
    context: str | None = None,
) -> dict[str, np.ndarray]:
    """Forecast with The Forecasting Company's pretrained t0, zero-shot.

    Like ``ChronosForecast``: the model reads the stream's recent history, and
    the ``exog`` streams as covariates known over the history and the horizon,
    and forecasts with quantiles from its own predictive distribution; the
    forecast's value is the median. t0-alpha (about 100M parameters) is trained
    on the 10, 25, 50, 75 and 90 % quantiles; others are interpolated between
    them, or extrapolated beyond them. It runs on the CPU and gives the same
    result every run. The weights (about 0.4 GB) are downloaded from Hugging
    Face the first time and cached. Needs the ``[t0]`` extra, which can't be
    installed alongside ``[pypots]``.

    Args:
        history: The stream (first column) and the exog streams, up to the
            issue time; given by flex-signal.
        future: The exog streams over the forecast rows; given by flex-signal.
        model: The pretrained model, by Hugging Face id:
            ``theforecastingcompany/t0-alpha`` (Apache-2.0).
        revision: The model's Hugging Face commit. Default: the commit this
            version of flex-signal pins for the model, so results don't change
            when the model is updated upstream.
        context: How much history the model reads, as a duration (e.g. ``14D``).
            Default: all of it.

    Returns:
        ``{"mean": ..., "q01": ..., ..., "q99": ...}``: the median and every
        whole-percent quantile over the forecast rows.
    """
    revision = _revision(model, revision, T0_METHOD)
    step, steps = _grid(history, future, T0_METHOD)
    window = _context(history, step, context, T0_METHOD)
    forecaster = _t0(model, revision)
    import torch

    target = window.iloc[:, 0].to_numpy(dtype="float32")
    kwargs: dict[str, Any] = {}
    if window.shape[1] > 1:
        exog = pd.concat([window.iloc[:, 1:], future[window.columns[1:]]])
        covariates = exog.to_numpy(dtype="float32").T[None]  # (1, F, T + H)
        kwargs["future_covariates"] = torch.from_numpy(covariates.copy())
    with seeded(0), torch.inference_mode():
        out = forecaster.predict(
            torch.from_numpy(target)[None],
            horizon=steps,
            quantile_levels=_LEVELS,
            **kwargs,
        )
    q = out.quantiles[0].to(torch.float64).numpy()  # (steps, levels)
    result = {quantile_key(level): q[:, i] for i, level in enumerate(_LEVELS)}
    result["mean"] = result["q50"]
    return result


register_forecaster(METHOD, chronos_forecast)
register_forecaster(T0_METHOD, t0_forecast)

register_backend(
    "chronos:chronos-2",
    "chronos:Chronos2Pipeline",
    extra=EXTRA,
    description="Chronos-2 zero-shot forecaster with covariates (Apache-2.0); "
    "fs.forecast.ChronosForecast.",
    docs_url="https://huggingface.co/amazon/chronos-2",
)
register_backend(
    "chronos:chronos-2-small",
    "chronos:Chronos2Pipeline",
    extra=EXTRA,
    description="Chronos-2 small: faster zero-shot forecaster (Apache-2.0); "
    "fs.forecast.ChronosForecast(model='autogluon/chronos-2-small').",
    docs_url="https://huggingface.co/autogluon/chronos-2-small",
)
register_backend(
    "timesfm:timesfm-2.5",
    "timesfm:TimesFM_2p5_200M_torch",
    extra=EXTRA,
    description="TimesFM 2.5 zero-shot forecaster (Apache-2.0; needs "
    "`pip install timesfm`).",
    docs_url="https://huggingface.co/google/timesfm-2.5-200m-pytorch",
)
register_backend(
    "t0:t0-alpha",
    "t0:T0Forecaster",
    extra=T0_EXTRA,
    description="t0-alpha zero-shot forecaster with covariates (Apache-2.0); "
    "fs.forecast.T0Forecast. Not installable alongside [pypots].",
    docs_url="https://huggingface.co/theforecastingcompany/t0-alpha",
)
