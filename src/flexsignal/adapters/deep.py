"""Neural-network forecasters from neuralforecast (the ``[deep]`` extra).

``fs.forecast.NeuralforecastForecast`` trains one of neuralforecast's models
(NHITS by default) on the stream's history each time it forecasts, then
forecasts the horizon. Trained weights aren't JSON, so nothing is stored: the
spec holds the model's settings and a seed, and the same data gives the same
forecast on the CPU.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    ModelName,
    model_class,
    model_option,
    regular_step,
    require,
    seed_torch,
    steps_in,
)
from flexsignal.backends import register_backend
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.steps.registered import register_forecaster

__all__ = ["MODELS", "QUANTILES", "neuralforecast_forecast"]

EXTRA = "deep"
METHOD = "NeuralforecastForecast"

MODELS: dict[str, str] = {
    "NHITS": "Hierarchical interpolation MLP; fast and accurate on seasonal data.",
    "NBEATS": "Basis-expansion MLP, univariate (no exog).",
    "NBEATSx": "NBEATS with exogenous inputs.",
    "MLP": "A plain multilayer perceptron; the fastest.",
    "LSTM": "A recurrent network.",
    "TiDE": "Time-series dense encoder; uses exog well.",
    "PatchTST": "Patch transformer, univariate (no exog); slower.",
    "DLinear": "Trend + seasonal linear model, univariate (no exog).",
}
"""The models flex-signal is tested with, with one line on each. Any other
neuralforecast model runs too, by its class name."""

OPTION = model_option(
    method=METHOD,
    option="model",
    library="neuralforecast",
    tested=tuple(MODELS),
    catalogue="https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html",
    note="Multivariate models get n_series=1, since one stream is forecast.",
)

QUANTILES = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
"""The quantiles ``loss='quantile'`` trains; a step may request any of them."""

# Constructor arguments the step sets itself, so ``params`` can't.
_OWNED = frozenset(
    {
        "h",
        "input_size",
        "max_steps",
        "random_seed",
        "loss",
        "valid_loss",
        "futr_exog_list",
        "hist_exog_list",
        "stat_exog_list",
        "scaler_type",
        "val_check_steps",
        "early_stop_patience_steps",
        "accelerator",
        "devices",
        "logger",
        "enable_progress_bar",
        "enable_model_summary",
        "enable_checkpointing",
        "callbacks",
        "trainer_kwargs",
    }
)
_LIGHTNING_LOGGERS = (
    "pytorch_lightning",
    "lightning",
    "lightning.pytorch",
    "lightning_fabric",
)
_LEVEL = re.compile(r"-(lo|hi)-([\d.]+)$")


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence lightning's logging and its known deprecation noise while training."""
    loggers = [logging.getLogger(name) for name in _LIGHTNING_LOGGERS]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*LeafSpec.*is deprecated")
            # pandas 3 hands out read-only arrays; neuralforecast copies them.
            warnings.filterwarnings(
                "ignore", message=r"The given NumPy array is not writable"
            )
            warnings.filterwarnings(
                "ignore", message=r".*does not have many workers", category=UserWarning
            )
            yield
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.setLevel(level)


def _check_params(cls: type, params: dict[str, Any], exog: list[str]) -> dict:
    if not isinstance(params, dict):
        raise SignalConfigError(
            f"{METHOD}: params must be a dict of extra {cls.__name__} arguments.",
            field="params",
        )
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "kwargs"}
    for key, value in params.items():
        if key in _OWNED:
            raise SignalConfigError(
                f"{METHOD}: params can't set {key!r}; flex-signal sets it (use the "
                "step's own options: horizon, input_size, max_steps, seed, loss, "
                "scaler, exog).",
                field=f"params.{key}",
            )
        if key not in accepted:
            raise SignalConfigError(
                f"{METHOD}: {cls.__name__} has no argument {key!r}. It accepts: "
                f"{', '.join(sorted(accepted - _OWNED))}.",
                field=f"params.{key}",
            )
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise SignalConfigError(
                f"{METHOD}: params.{key} must be a JSON value, not {value!r}.",
                field=f"params.{key}",
            ) from None
    if exog and not getattr(cls, "EXOGENOUS_FUTR", False):
        with_exog = [m for m in MODELS if m not in ("NBEATS", "PatchTST", "DLinear")]
        raise SignalConfigError(
            f"{METHOD}: {cls.__name__} can't use exog streams; drop exog or pick a "
            f"model that can: {', '.join(with_exog)}.",
            field="model",
        )
    return dict(params)


def _frames(
    history: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert to neuralforecast's long format: ``unique_id, ds, y`` (+ exog)."""
    exog = [str(c) for c in history.columns[1:]]
    y = history.iloc[:, 0]
    first = y.first_valid_index()
    history = history.loc[first:]
    for col in exog:
        if history[col].isna().any():
            when = history.index[history[col].isna().to_numpy()][0]
            raise SignalDataError(
                f"{METHOD}: exog stream {col!r} has no value at {when} in the "
                "history; fill it (e.g. an impute step) before the forecast step."
            )
    ds = history.index.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame(
        {
            "unique_id": "y",
            "ds": ds,
            "y": history.iloc[:, 0].fillna(0.0).to_numpy("float64"),
            "available_mask": history.iloc[:, 0].notna().to_numpy("float64"),
        }
    )
    for col in exog:
        df[col] = history[col].to_numpy("float64")
    if not exog:
        return df, None
    futr = pd.DataFrame(
        {"unique_id": "y", "ds": future.index.tz_convert("UTC").tz_localize(None)}
    )
    for col in exog:
        futr[col] = future[col].to_numpy("float64")
    return df, futr


def _quantile_of(column: str) -> float | None:
    if column.endswith("-median"):
        return 0.5
    match = _LEVEL.search(column)
    if match is None:
        return None
    half = float(match.group(2)) / 200
    return round(0.5 - half if match.group(1) == "lo" else 0.5 + half, 4)


def neuralforecast_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: ModelName = "NHITS",
    input_size: str | None = None,
    max_steps: int = 300,
    seed: int = 0,
    loss: Literal["quantile", "mae", "mse"] = "quantile",
    scaler: Literal["standard", "robust", "minmax", "identity"] = "standard",
    params: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Train a neural network on the history, then forecast the horizon.

    Each forecast trains a fresh neuralforecast model on the stream's history
    (and the ``exog`` streams, known into the future) and forecasts the next
    ``horizon``. Use it with plenty of history (weeks of data, many times
    ``input_size``) and strong repeating patterns; with little data a
    statistical model or ``ChronosForecast`` is usually better. Training runs on
    the CPU, takes seconds to minutes, and is repeatable for a given ``seed``.
    Missing values in the history are masked out of the training loss.

    Args:
        history: The stream (first column) and the exog streams, up to the
            issue time; given by flex-signal.
        future: The exog streams over the forecast rows; given by flex-signal.
        model: The network, by class name. Tested: ``NHITS`` (default),
            ``NBEATS``, ``NBEATSx``, ``MLP``, ``LSTM``, ``TiDE``, ``PatchTST``
            and ``DLinear``; NBEATS, PatchTST and DLinear don't use exog
            streams. Any other neuralforecast model, e.g. ``TFT`` or
            ``iTransformer``, runs too, untested; its own settings go in
            ``params``.
        input_size: How much history the network looks at for each forecast, as
            a duration (e.g. ``2D``). Default: twice the horizon.
        max_steps: How many training steps (batches) to run. More fits better
            but takes longer.
        seed: The random seed, so training repeats exactly.
        loss: ``quantile`` (default) trains on a set of quantiles (1%, 5%, 10%,
            20%, ..., 90%, 95%, 99%), so the step can write quantile streams;
            the forecast is the median. ``mae`` or ``mse`` train a point
            forecast only (no quantiles).
        scaler: How each training window is scaled before the network sees it:
            ``standard`` (default), ``robust``, ``minmax`` or ``identity``.
        params: Extra arguments for the model's constructor, e.g.
            ``{"learning_rate": 0.001, "batch_size": 32}``. Arguments the step
            sets itself (such as ``h`` or ``loss``) are refused.

    Returns:
        ``{"mean": ...}``, plus ``"q01"`` ... ``"q99"`` for the trained
        quantiles with ``loss='quantile'``.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise SignalConfigError(
            f"{METHOD}: max_steps must be a whole number of at least 1, e.g. 300.",
            field="max_steps",
            value=max_steps,
        )
    nf = require("neuralforecast", EXTRA, method=METHOD)
    models = require("neuralforecast.models", EXTRA, method=METHOD)
    losses = require("neuralforecast.losses.pytorch", EXTRA, method=METHOD)
    base = require("neuralforecast.common._base_model", EXTRA, method=METHOD)
    cls = model_class(models, model, OPTION, base=base.BaseModel)
    exog = [str(c) for c in history.columns[1:]]
    extra = _check_params(cls, params or {}, exog)

    step = regular_step(history.index, METHOD)
    h = len(future)
    if h == 0 or not future.index.equals(
        pd.date_range(history.index[-1] + step, periods=h, freq=step)
    ):
        raise SignalDataError(
            f"{METHOD}: the forecast rows must follow the history one data step "
            f"({step}) apart."
        )
    n_input = (
        2 * h
        if input_size is None
        else steps_in(input_size, step, field="input_size", method=METHOD)
    )
    df, futr = _frames(history, future)
    if len(df) < n_input + h:
        raise SignalDataError(
            f"{METHOD}: {history.columns[0]!r} has {len(df)} steps of history, but "
            f"training needs at least input_size + horizon = {n_input + h}; use a "
            "shorter input_size or more history."
        )

    if loss == "quantile":
        train_loss = losses.MQLoss(quantiles=list(QUANTILES))
    elif loss == "mae":
        train_loss = losses.MAE()
    else:
        train_loss = losses.MSE()
    kwargs: dict[str, Any] = {
        "h": h,
        "input_size": n_input,
        "max_steps": max_steps,
        "val_check_steps": max_steps,
        "random_seed": seed,
        "loss": train_loss,
        "scaler_type": scaler,
        "accelerator": "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        **extra,
    }
    if exog:
        kwargs["futr_exog_list"] = exog
    signature = inspect.signature(cls.__init__).parameters
    if "n_series" in signature and "n_series" not in kwargs:
        kwargs["n_series"] = 1  # a multivariate model, given one stream
    missing = [
        name
        for name, p in signature.items()
        if name != "self"
        and p.default is p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and name not in kwargs
    ]
    if missing:
        raise SignalConfigError(
            f"{METHOD}: neuralforecast {model} needs {missing}: give them in "
            f"params, e.g. params={{{missing[0]!r}: ...}}. See {OPTION.catalogue}.",
            field="params",
            value=missing,
        )
    with _quiet():
        seed_torch(seed)
        network = cls(**kwargs)
        freq = pd.tseries.frequencies.to_offset(step).freqstr
        forecaster = nf.NeuralForecast(models=[network], freq=freq)
        forecaster.fit(df)
        out = forecaster.predict(futr_df=futr)
    if len(out) != h:
        raise SignalDataError(
            f"{METHOD}: {model} returned {len(out)} rows for a {h}-step horizon."
        )
    columns = [c for c in out.columns if c.startswith(cls.__name__)]
    if loss != "quantile":
        return {"mean": out[columns[0]].to_numpy("float64")}
    result: dict[str, np.ndarray] = {}
    for col in columns:
        q = _quantile_of(col)
        if q is not None:
            result[f"q{round(q * 100):02d}"] = out[col].to_numpy("float64")
    result["mean"] = result["q50"]
    return result


register_forecaster(METHOD, neuralforecast_forecast)

for _name, _description in MODELS.items():
    register_backend(
        f"neuralforecast:{_name}",
        f"neuralforecast.models:{_name}",
        extra=EXTRA,
        description=f"{_description} fs.forecast.NeuralforecastForecast"
        f"(model={_name!r}).",
        docs_url="https://nixtlaverse.nixtla.io/neuralforecast/",
    )
