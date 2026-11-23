"""PyPOTS deep-learning models as an imputer and a forecaster (extra ``pypots``).

`PyPOTS <https://docs.pypots.com>`_ collects neural networks for time series
with missing values. This module registers two methods:

- ``PypotsImpute``: fills gaps with an imputation network (SAITS, BRITS, CSDI or
  TimesNet) trained on the data it is filling. It can use other streams as extra
  inputs, so a gap in one meter can be filled from its neighbours.
- ``PypotsForecast``: forecasts a stream from its own recent history with a
  forecasting network (DLinear, TimesNet, Transformer, TEFN or CSDI). CSDI
  samples a distribution, so it alone can write quantile streams.

Trained weights can't be saved as JSON, so each model is trained, on CPU and
from a fixed seed, every time the step runs, and nothing is stored. Use a small
model and few epochs on long data; a GPU is not used.

The windowing functions :func:`to_windows` and :func:`from_windows` turn a
DataFrame into PyPOTS' ``[windows, steps, features]`` arrays and back, and are
useful on their own for running other PyPOTS models.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    Forecast,
    ModelName,
    build_model,
    model_class,
    model_option,
    quantile_key,
    regular_step,
    require,
    seeded,
    steps_in,
)
from flexsignal.backends import register_backend
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.steps.registered import register_forecaster, register_imputer

__all__ = [
    "IMPUTE_MODELS",
    "FORECAST_MODELS",
    "from_windows",
    "pypots_forecast",
    "pypots_impute",
    "to_windows",
    "window_starts",
]

EXTRA = "pypots"

IMPUTE_MODELS: tuple[str, ...] = ("SAITS", "BRITS", "CSDI", "TimesNet")
"""The PyPOTS imputation models ``PypotsImpute`` is tested with. Any other
PyPOTS imputation model runs too, with its own settings in ``params``."""

FORECAST_MODELS: tuple[str, ...] = (
    "DLinear",
    "TimesNet",
    "Transformer",
    "TEFN",
    "CSDI",
)
"""The PyPOTS forecasting models ``PypotsForecast`` is tested with (all on CPU).
Any other PyPOTS forecasting model runs too, with its own settings in
``params``."""

IMPUTE_OPTION = model_option(
    method="PypotsImpute",
    option="model",
    library="PyPOTS",
    tested=IMPUTE_MODELS,
    catalogue="https://docs.pypots.com/en/latest/pypots.imputation.html",
)
FORECAST_OPTION = model_option(
    method="PypotsForecast",
    option="model",
    library="PyPOTS",
    tested=FORECAST_MODELS,
    catalogue="https://docs.pypots.com/en/latest/pypots.forecasting.html",
    note="Only models that sample a distribution (CSDI) can write quantiles.",
)

# Constructor arguments the adapter sets itself; ``params`` may not set them.
_MANAGED = frozenset(
    {
        "n_steps",
        "n_features",
        "n_pred_steps",
        "n_pred_features",
        "epochs",
        "batch_size",
        "optimizer",
        "device",
        "saving_path",
        "model_saving_strategy",
        "verbose",
        "num_workers",
        "training_loss",
        "validation_metric",
        "patience",
    }
)

_DOCS = "https://docs.pypots.com/en/latest/"


# -- windowing ------------------------------------------------------------------------


def window_starts(n: int, window: int, stride: int | None = None) -> np.ndarray:
    """Return where each window starts when ``n`` rows are cut into windows.

    Windows start at 0, ``stride``, ``2 * stride``, ... If the last of those
    doesn't reach the end, one more window is added that ends on the last row,
    so every row is in at least one window. With fewer rows than ``window``
    there is one window, padded with NaN after the last row.

    Args:
        n: Number of rows.
        window: Rows per window.
        stride: Rows between window starts. Default: ``window`` (no overlap).

    Returns:
        The start positions, in order.

    Raises:
        SignalConfigError: If ``window`` or ``stride`` is below 1.
    """
    stride = window if stride is None else stride
    if int(window) != window or window < 1:
        raise SignalConfigError(
            f"window must be a whole number of rows, at least 1; got {window}.",
            field="window",
        )
    if int(stride) != stride or stride < 1:
        raise SignalConfigError(
            f"stride must be a whole number of rows, at least 1; got {stride}.",
            field="stride",
        )
    if n < 1:
        raise SignalDataError("there are no rows to cut into windows.")
    if n <= window:
        return np.array([0])
    starts = np.arange(0, n - window + 1, stride)
    if starts[-1] + window < n:
        starts = np.append(starts, n - window)
    return starts


def to_windows(
    frame: pd.DataFrame, window: int, *, stride: int | None = None
) -> dict[str, np.ndarray]:
    """Cut a DataFrame into the windows a PyPOTS model takes as input.

    Use it to run any PyPOTS model on flex-signal data; :func:`from_windows`
    turns the model's output back into a DataFrame on the same index.

    Args:
        frame: Streams as columns on a regular time index.
        window: Rows per window (the model's ``n_steps``).
        stride: Rows between window starts. Default: ``window``, so windows
            don't overlap. A smaller stride gives more, overlapping windows;
            :func:`from_windows` averages where they overlap.

    Returns:
        ``{"X": array}`` of shape ``[n_windows, window, n_columns]``, float64,
        NaN where a value is missing (and after the last row, if the frame is
        shorter than one window).

    Raises:
        SignalDataError: If the index isn't regular, or a column isn't numeric.
    """
    regular_step(frame.index, "to_windows")
    try:
        values = frame.to_numpy(dtype="float64", na_value=np.nan)
    except (TypeError, ValueError):
        raise SignalDataError("to_windows needs numeric columns.") from None
    starts = window_starts(len(frame), window, stride)
    if len(frame) < window:
        pad = np.full((window - len(frame), values.shape[1]), np.nan)
        values = np.vstack([values, pad])
    X = np.stack([values[s : s + window] for s in starts])
    return {"X": X}


def from_windows(
    windows: np.ndarray,
    index: pd.DatetimeIndex,
    columns: Sequence[str],
    *,
    stride: int | None = None,
) -> pd.DataFrame:
    """Put a model's windowed output back on the original time index.

    Where windows overlap, their values are averaged (ignoring NaN). Padding
    after the last row is dropped.

    Args:
        windows: Array of shape ``[n_windows, window, n_columns]``, from a model
            run on :func:`to_windows` output.
        index: The index the windows were cut from.
        columns: Column names, in the windows' feature order.
        stride: The stride given to :func:`to_windows`.

    Returns:
        A DataFrame on ``index`` with ``columns``.

    Raises:
        SignalDataError: If the array's shape doesn't match the index, window
            and stride.
    """
    windows = np.asarray(windows, dtype="float64")
    if windows.ndim != 3:
        raise SignalDataError(
            f"from_windows needs a [windows, steps, features] array, got shape "
            f"{windows.shape}."
        )
    n_windows, window, n_features = windows.shape
    starts = window_starts(len(index), window, stride)
    if n_windows != len(starts) or n_features != len(columns):
        raise SignalDataError(
            f"from_windows got {n_windows} windows of {n_features} features; "
            f"{len(index)} rows in windows of {window} make {len(starts)} windows, "
            f"and there are {len(columns)} columns."
        )
    n = max(len(index), window)
    total = np.zeros((n, n_features))
    count = np.zeros((n, n_features))
    for start, block in zip(starts, windows, strict=True):
        seen = ~np.isnan(block)
        total[start : start + window] += np.where(seen, block, 0.0)
        count[start : start + window] += seen
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return pd.DataFrame(mean[: len(index)], index=index, columns=list(columns))


# -- shared training helpers ----------------------------------------------------------


class _Errors(logging.Handler):
    """Collect the error messages PyPOTS logs, so they can be put in our error."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.messages.append(record.getMessage())


@contextmanager
def _quiet(method: str) -> Iterator[_Errors]:
    """Silence PyPOTS' log and its known harmless warnings; keep its errors."""
    log = require("pypots.utils.logging", EXTRA, method=method).logger
    saved = log.level, log.handlers[:], log.propagate
    levels = [h.level for h in log.handlers]
    errors = _Errors()
    log.handlers = [errors]
    log.propagate = False
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="enable_nested_tensor is True", category=UserWarning
            )
            yield errors
    finally:
        log.setLevel(saved[0])
        log.handlers = saved[1]
        log.propagate = saved[2]
        for handler, level in zip(saved[1], levels, strict=True):
            handler.setLevel(level)


def _check_options(
    method: str,
    *,
    epochs: int,
    batch_size: int,
    n_layers: int,
    d_model: int,
    n_heads: int,
    learning_rate: float,
    samples: int,
) -> None:
    for field, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("n_layers", n_layers),
        ("d_model", d_model),
        ("n_heads", n_heads),
        ("samples", samples),
    ):
        if value < 1:
            raise SignalConfigError(
                f"{method}: {field} must be at least 1, got {value}.",
                field=field,
                value=value,
            )
    if d_model % n_heads:
        raise SignalConfigError(
            f"{method}: d_model ({d_model}) must be a multiple of n_heads "
            f"({n_heads}), e.g. d_model=32 with n_heads=4.",
            field="d_model",
            value=d_model,
        )
    if not learning_rate > 0:
        raise SignalConfigError(
            f"{method}: learning_rate must be above 0, got {learning_rate}.",
            field="learning_rate",
            value=learning_rate,
        )


def _common(method: str, *, epochs: int, batch_size: int, learning_rate: float):
    optim = require("pypots.optim", EXTRA, method=method)
    # A fresh optimizer each time: some PyPOTS models share one default instance.
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optim.Adam(lr=learning_rate),
        "device": "cpu",
        "saving_path": None,
        "model_saving_strategy": None,
        "verbose": False,
    }


def _train(method: str, model: Any, train: dict[str, np.ndarray], errors: _Errors):
    try:
        model.fit(train)
    except RuntimeError as exc:
        detail = "; ".join(errors.messages) or str(exc)
        raise SignalDataError(
            f"{method}: training {type(model).__name__} failed: {detail}. Check "
            "that the window is long enough for the model and that the data has "
            "values."
        ) from None


def _scale(block: np.ndarray, method: str, names: Sequence[str]):
    """Standardize each column by its observed values."""
    seen = ~np.isnan(block)
    empty = [
        name
        for name, any_seen in zip(names, seen.any(axis=0), strict=True)
        if not any_seen
    ]
    if empty:
        raise SignalDataError(f"{method}: {empty} have no values to learn from.")
    mean = np.nanmean(block, axis=0)
    std = np.nanstd(block, axis=0)
    std = np.where(std > 0, std, 1.0)
    return (block - mean) / std, mean, std


def _durations(
    index: pd.DatetimeIndex, method: str, window: str, stride: str | None
) -> tuple[int, int | None]:
    step = regular_step(index, method)
    n_window = steps_in(window, step, field="window", method=method)
    n_stride = (
        None
        if stride is None
        else steps_in(stride, step, field="stride", method=method)
    )
    return n_window, n_stride


# -- the imputer ----------------------------------------------------------------------


def _impute_sizes(
    model: str, *, n_layers: int, d_model: int, n_heads: int, window: int
) -> dict[str, Any]:
    if model == "SAITS":
        return {
            "n_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_k": d_model // n_heads,
            "d_v": d_model // n_heads,
            "d_ffn": 2 * d_model,
        }
    if model == "BRITS":
        return {"rnn_hidden_size": d_model}
    if model == "CSDI":
        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_channels": d_model,
            "d_time_embedding": 2 * d_model,
            "d_feature_embedding": 16,
            "d_diffusion_embedding": 2 * d_model,
        }
    if model == "TimesNet":
        return {
            "n_layers": n_layers,
            "top_k": max(1, min(3, window // 2)),
            "d_model": d_model,
            "d_ffn": 2 * d_model,
            "n_kernels": 3,
        }
    # An untested model gets the common sizes it takes; the rest come from params.
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


def pypots_impute(
    values: pd.DataFrame,
    gaps: pd.Series,
    *,
    model: ModelName = "SAITS",
    window: str = "1D",
    stride: str | None = None,
    features: list[str] = [],  # noqa: B006 - an option default, copied by pydantic
    epochs: int = 30,
    batch_size: int = 32,
    n_layers: int = 2,
    d_model: int = 32,
    n_heads: int = 4,
    learning_rate: float = 0.001,
    samples: int = 10,
    seed: int = 0,
    params: dict[str, Any] = {},  # noqa: B006 - an option default, copied by pydantic
) -> np.ndarray:
    """Fill gaps with a PyPOTS imputation network trained on the data itself.

    The stream, and any ``features`` streams, are standardized and cut into
    windows; the network learns to rebuild values PyPOTS hides from it in
    those windows, then fills the real gaps. Use it for long or irregular gaps
    in streams with repeating patterns (daily cycles) or with related streams,
    when interpolation and regression fall short. It trains every time it runs,
    so keep the model small; on short data, prefer ``kalman`` or
    ``regression``.

    Args:
        values: Every stream; the one being filled first.
        gaps: True where a value may be filled.
        model: The PyPOTS network, by class name. Tested: ``SAITS``
            (self-attention, a strong default), ``BRITS`` (bidirectional RNN),
            ``CSDI`` (diffusion; slow, needs many training batches before its
            fills are usable, averages ``samples`` draws) and ``TimesNet``
            (convolutions over periods). Any other PyPOTS imputation model,
            e.g. ``ImputeFormer``, runs too, untested: give its own required
            settings in ``params``.
        window: Length of the windows the model sees, e.g. 1D. Make it longer
            than the longest gap and at least one cycle of the data.
        stride: Time between window starts, e.g. 6h. Default: half the window.
            Overlapping fills are averaged.
        features: Other streams fed to the model as extra inputs, e.g.
            ["pressure"]. Default: none.
        epochs: Training passes over the windows.
        batch_size: Windows per training batch.
        n_layers: Network layers (ignored by BRITS).
        d_model: Width of the network: SAITS/TimesNet model size, BRITS hidden
            size, CSDI channels.
        n_heads: Attention heads for SAITS and CSDI; d_model must be a multiple
            of it.
        learning_rate: The Adam optimizer's learning rate.
        samples: CSDI only: how many draws to average into each fill.
        seed: Seed for the weights and training, so runs repeat exactly.
        params: Other arguments of the PyPOTS model's constructor, e.g.
            {"dropout": 0.1} for SAITS, or the required settings of an untested
            model; checked against the model when it runs.

    Returns:
        Proposed values for the whole stream.

    Raises:
        SignalConfigError: If an option is invalid.
        SignalDataError: If the data is irregular, shorter than one window, or
            a stream has no values.
    """
    method = "PypotsImpute"
    _check_options(
        method,
        epochs=epochs,
        batch_size=batch_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        learning_rate=learning_rate,
        samples=samples,
    )
    target = values.columns[0]
    bad = [f for f in features if f not in values.columns or f == target]
    if bad:
        raise SignalConfigError(
            f"{method}: features {bad} must be other streams in the data, not "
            f"{target!r} itself; the data has {list(values.columns)}.",
            field="features",
            value=bad,
        )
    columns = [target, *dict.fromkeys(features)]
    n_window, n_stride = _durations(values.index, method, window, stride)
    if len(values) < n_window:
        raise SignalDataError(
            f"{method}: the data has {len(values)} rows, fewer than one {window} "
            "window; use a shorter window."
        )
    if n_stride is None:
        n_stride = max(1, n_window // 2)

    imputation = require("pypots.imputation", EXTRA, method=method)
    bases = require("pypots.imputation.base", EXTRA, method=method)
    base = (bases.BaseImputer, bases.BaseNNImputer)
    cls = model_class(imputation, model, IMPUTE_OPTION, base=base)
    block = values[columns].to_numpy(dtype="float64", na_value=np.nan)
    scaled, mean, std = _scale(block, method, columns)
    frame = pd.DataFrame(scaled, index=values.index, columns=columns)
    X = to_windows(frame, n_window, stride=n_stride)["X"].astype("float32")

    with seeded(seed), _quiet(method) as errors:
        net = build_model(
            cls,
            {
                "n_steps": n_window,
                "n_features": len(columns),
                **_impute_sizes(
                    model,
                    n_layers=n_layers,
                    d_model=d_model,
                    n_heads=n_heads,
                    window=n_window,
                ),
                **_common(
                    method,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                ),
            },
            params,
            IMPUTE_OPTION,
            managed=_MANAGED,
        )
        _train(method, net, {"X": X}, errors)
        kwargs = {"n_sampling_times": samples} if model == "CSDI" else {}
        out = np.asarray(net.predict({"X": X}, **kwargs)["imputation"])
    if out.ndim == 4:  # CSDI: [windows, samples, steps, features]
        out = out.mean(axis=1)
    filled = from_windows(out, values.index, columns, stride=n_stride)
    return filled[target].to_numpy() * std[0] + mean[0]


# -- the forecaster -------------------------------------------------------------------


def _forecast_sizes(
    model: str,
    *,
    n_layers: int,
    d_model: int,
    n_heads: int,
    lookback: int,
) -> dict[str, Any]:
    if model == "DLinear":
        size = min(25, lookback)
        return {
            "moving_avg_window_size": size if size % 2 else size - 1 or 1,
            "d_model": d_model,
        }
    if model == "TimesNet":
        return {
            "n_layers": n_layers,
            "top_k": max(1, min(3, lookback // 2)),
            "d_model": d_model,
            "d_ffn": 2 * d_model,
            "n_kernels": 3,
        }
    if model == "Transformer":
        return {
            "n_encoder_layers": n_layers,
            "n_decoder_layers": n_layers,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_k": d_model // n_heads,
            "d_v": d_model // n_heads,
            "d_ffn": 2 * d_model,
        }
    if model == "TEFN":
        return {}
    if model == "CSDI":
        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_channels": d_model,
            "d_time_embedding": 2 * d_model,
            "d_feature_embedding": 16,
            "d_diffusion_embedding": 2 * d_model,
        }
    # An untested model gets the common sizes it takes; the rest come from params.
    return {"n_layers": n_layers, "d_model": d_model, "n_heads": n_heads}


def pypots_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    model: ModelName = "DLinear",
    lookback: str = "1D",
    stride: str | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    n_layers: int = 2,
    d_model: int = 32,
    n_heads: int = 4,
    learning_rate: float = 0.001,
    samples: int = 50,
    seed: int = 0,
    params: dict[str, Any] = {},  # noqa: B006 - an option default, copied by pydantic
) -> np.ndarray | dict[str, np.ndarray]:
    """Forecast a stream from its own history with a PyPOTS forecasting network.

    The history is standardized and cut into training pairs: ``lookback`` of
    past values and the ``horizon`` after them. The network learns to map one
    to the other, then forecasts the horizon from the last ``lookback`` before
    the issue time. Use it on long histories with repeating patterns. It
    trains every time it runs (each backtest fold too), and it uses only the
    target's own past: PyPOTS' forecasting models here take no known-ahead
    inputs, so ``exog`` is refused. Only ``CSDI`` samples a distribution, so
    only it can write ``quantiles``.

    Args:
        history: The target's values up to the issue time.
        future: The forecast rows (no columns: exog is refused).
        model: The PyPOTS network, by class name. Tested: ``DLinear`` (trend
            + seasonal linear; fast, a strong default), ``TimesNet``
            (convolutions over periods), ``Transformer``, ``TEFN`` (small
            evidence-based network) and ``CSDI`` (diffusion; slow; gives
            quantiles from ``samples`` draws). Any other PyPOTS forecasting
            model, e.g. ``SegRNN``, runs too, untested: give its own required
            settings in ``params``.
        lookback: How much history each forecast is made from, e.g. 1D. At
            least one cycle of the data.
        stride: Time between the starts of the training pairs, e.g. 1h.
            Default: one data step (every possible pair).
        epochs: Training passes over the pairs.
        batch_size: Pairs per training batch.
        n_layers: Network layers (ignored by DLinear and TEFN).
        d_model: Width of the network (CSDI: channels; ignored by TEFN).
        n_heads: Attention heads for Transformer and CSDI; d_model must be a
            multiple of it.
        learning_rate: The Adam optimizer's learning rate.
        samples: CSDI only: draws whose mean is the forecast and whose spread
            gives the quantiles. Use 50 or more for 10%/90% quantiles.
        seed: Seed for the weights and training, so runs repeat exactly.
        params: Other arguments of the PyPOTS model's constructor, e.g.
            {"dropout": 0.1} for Transformer, or the required settings of an
            untested model; checked against the model when it runs.

    Returns:
        The mean forecast; for CSDI, also the quantiles ``q01``-``q99``.

    Raises:
        SignalConfigError: If an option is invalid, ``exog`` is set, or
            quantiles are requested from a model other than CSDI.
        SignalDataError: If the history is irregular or shorter than
            ``lookback`` plus the horizon.
    """
    method = "PypotsForecast"
    _check_options(
        method,
        epochs=epochs,
        batch_size=batch_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        learning_rate=learning_rate,
        samples=samples,
    )
    if history.shape[1] > 1:
        raise SignalConfigError(
            f"{method} forecasts from the target's own history; PyPOTS' forecasting "
            f"models take no known-ahead inputs, so exog {list(history.columns[1:])} "
            "can't be used. Remove exog, or use forecast.Regression or "
            "forecast.SARIMAX.",
            field="exog",
        )
    n_lookback, n_stride = _durations(history.index, method, lookback, stride)
    horizon = len(future)
    y = history.iloc[:, 0].to_numpy(dtype="float64", na_value=np.nan)
    if len(y) < n_lookback + horizon:
        raise SignalDataError(
            f"{method}: the history has {len(y)} rows; it needs at least lookback "
            f"({n_lookback}) + horizon ({horizon}) to learn from. Use a shorter "
            "lookback or more history."
        )
    scaled, mean, std = _scale(y[:, None], method, [history.columns[0]])
    pairs = np.lib.stride_tricks.sliding_window_view(
        scaled[:, 0], n_lookback + horizon
    )[:: n_stride or 1]
    seen = ~np.isnan(pairs)
    keep = seen[:, :n_lookback].any(axis=1) & seen[:, n_lookback:].any(axis=1)
    if not keep.any():
        raise SignalDataError(
            f"{method}: no stretch of {n_lookback + horizon} rows has values both "
            "before and after its forecast point, so there is nothing to learn from."
        )
    pairs = pairs[keep].astype("float32")
    train = {
        "X": pairs[:, :n_lookback, None],
        "X_pred": pairs[:, n_lookback:, None],
    }
    last = scaled[-n_lookback:, 0][None, :, None].astype("float32")

    forecasting = require("pypots.forecasting", EXTRA, method=method)
    bases = require("pypots.forecasting.base", EXTRA, method=method)
    base = (bases.BaseForecaster, bases.BaseNNForecaster)
    cls = model_class(forecasting, model, FORECAST_OPTION, base=base)
    with seeded(seed), _quiet(method) as errors:
        net = build_model(
            cls,
            {
                "n_steps": n_lookback,
                "n_features": 1,
                "n_pred_steps": horizon,
                "n_pred_features": 1,
                **_forecast_sizes(
                    model,
                    n_layers=n_layers,
                    d_model=d_model,
                    n_heads=n_heads,
                    lookback=n_lookback,
                ),
                **_common(
                    method,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                ),
            },
            params,
            FORECAST_OPTION,
            managed=_MANAGED,
        )
        _train(method, net, train, errors)
        if model == "CSDI":
            test = {"X": last, "X_pred": np.full((1, horizon, 1), np.nan, "float32")}
            out = net.predict(test, n_sampling_times=samples)["forecasting"]
        else:
            out = net.predict({"X": last})["forecasting"]
    out = np.asarray(out, dtype="float64") * std[0] + mean[0]
    if model != "CSDI":
        return Forecast(
            out.reshape(horizon),
            None,
            f"{method}: model {model!r} gives a single forecast, not a "
            "distribution, so it can't write quantiles. Use model='CSDI', which "
            "samples one, or remove quantiles.",
        )
    draws = out.reshape(samples, horizon)  # [1, samples, horizon, 1]
    result = {"mean": draws.mean(axis=0)}
    for percent in range(1, 100):
        q = percent / 100
        result[quantile_key(q)] = np.quantile(draws, q, axis=0)
    return result


# -- registration ---------------------------------------------------------------------

register_imputer("PypotsImpute", pypots_impute)
register_forecaster("PypotsForecast", pypots_forecast)

_DESCRIPTIONS = {
    "imputation.SAITS": "SAITS: self-attention imputation network.",
    "imputation.BRITS": "BRITS: bidirectional recurrent imputation network.",
    "imputation.CSDI": "CSDI: conditional score-based diffusion imputation.",
    "imputation.TimesNet": "TimesNet: 2-D convolutions over a series' periods.",
    "forecasting.DLinear": "DLinear: linear layers on trend and seasonal parts.",
    "forecasting.TimesNet": "TimesNet: 2-D convolutions over a series' periods.",
    "forecasting.Transformer": "Transformer encoder-decoder forecaster.",
    "forecasting.TEFN": "TEFN: time evidence fusion network.",
    "forecasting.CSDI": "CSDI: diffusion forecaster that samples a distribution.",
}
for _path, _description in _DESCRIPTIONS.items():
    _module, _name = _path.split(".")
    register_backend(
        f"pypots:{_path}",
        f"pypots.{_module}:{_name}",
        extra=EXTRA,
        description=_description,
        docs_url=f"{_DOCS}pypots.{_module}.html#pypots.{_module}.{_name}",
    )
