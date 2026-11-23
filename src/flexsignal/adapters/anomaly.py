"""Anomaly cleaners from the ``[anomaly]`` extra: pyod, ruptures and stumpy.

Three cleaning steps that find bad values with an anomaly-detection library and
reject them: flex-signal sets each rejected value to NaN and flags it
``OUTLIER``. Values that are already missing are skipped, never rejected.

- ``PyodDetector``: a pyod outlier detector (ECOD, Isolation Forest, ...)
  trained on the stream it cleans, on features built from the stream.
- ``RupturesChangepoint``: ruptures change-point detection; rejects short
  segments that jump away from the level on both sides (a transient glitch)
  and leaves long regimes alone.
- ``StumpyDiscord``: stumpy matrix-profile discords; rejects the samples of
  stretches unlike anything else in the stream.

Each method is trained on the data it is given, when the step runs, and keeps
no state (like ``knn`` and ``iterative``): a fitted pipeline
re-runs the method on new data. Results are deterministic.

Import this module to register the methods, e.g.
``fs.load_plugins("flexsignal.adapters.anomaly")``. Importing it doesn't import
pyod, ruptures or stumpy; each is imported when its step runs, and a missing one
raises an error naming the ``anomaly`` extra.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from flexsignal.adapters._common import (
    ModelName,
    build_model,
    model_class,
    model_option,
    regular_step,
    require,
    steps_in,
)
from flexsignal.backends import register_backend
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.steps.registered import register_cleaner

__all__ = ["pyod_detector", "ruptures_changepoint", "stumpy_discord"]

EXTRA = "anomaly"
"""The flex-signal extra that installs pyod, ruptures and stumpy."""

_MAD_TO_SIGMA = 1.4826
"""Turns a median absolute deviation into a standard deviation for normal data."""

_PYOD_MODELS = {
    "ECOD": ("pyod.models.ecod", "Empirical cumulative-distribution outlier score."),
    "COPOD": ("pyod.models.copod", "Copula-based outlier score; no parameters."),
    "HBOS": ("pyod.models.hbos", "Histogram-based outlier score; fast."),
    "IForest": ("pyod.models.iforest", "Isolation Forest; seeded."),
    "KNN": ("pyod.models.knn", "Distance to the k-th nearest neighbour."),
    "LOF": ("pyod.models.lof", "Local outlier factor (local density)."),
}
_PYOD_FEATURES = ["value", "residual"]
"""PyodDetector's default features (never mutated)."""
_PYOD_MIN_VALUES = 30
"""Fewest present values PyodDetector trains on (LOF uses 20 neighbours)."""

_RUPTURES_ALGORITHMS = {
    "Pelt": "Pruned exact linear time search for the best segmentation.",
    "Binseg": "Binary segmentation; approximate and fast.",
    "BottomUp": "Bottom-up segmentation; approximate and fast.",
}

PYOD_OPTION = model_option(
    method="PyodDetector",
    option="model",
    library="pyod",
    tested=tuple(_PYOD_MODELS),
    catalogue="https://pyod.readthedocs.io/en/latest/pyod.models.html",
)
RUPTURES_OPTION = model_option(
    method="RupturesChangepoint",
    option="algorithm",
    library="ruptures",
    tested=tuple(_RUPTURES_ALGORITHMS),
    catalogue="https://centre-borelli.github.io/ruptures-docs/user-guide/",
    note="The algorithm must take a penalty (pen); Dynp, which takes a number "
    "of change points instead, doesn't fit.",
)


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _pyod_modules(name: str, path: list[str]) -> list[str]:
    """The pyod.models modules that may define detector ``name``.

    First the modules named like it (``ecod`` for ECOD, ``feature_bagging`` for
    FeatureBagging); then any other module whose source defines the class. The
    source is only read, not imported: importing every pyod module is slow and
    prints install hints for the libraries some detectors need.
    """
    found = list(dict.fromkeys([name.lower(), _snake(name)]))
    defines = re.compile(rf"^class {re.escape(name)}", re.MULTILINE)
    for info in pkgutil.iter_modules(path):
        if info.name in found:
            continue
        folder = Path(getattr(info.module_finder, "path", ""))
        source = (
            folder / info.name / "__init__.py"
            if info.ispkg
            else folder / f"{info.name}.py"
        )
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if defines.search(text):
            found.append(info.name)
    return found


def _pyod_class(name: str, method: str) -> type:
    """Find a pyod detector by class name, in pyod.models' own modules."""
    if name in _PYOD_MODELS:
        return getattr(require(_PYOD_MODELS[name][0], EXTRA, method=method), name)
    models = require("pyod.models", EXTRA, method=method)
    base = require("pyod.models.base", EXTRA, method=method).BaseDetector
    for module_name in _pyod_modules(name, list(models.__path__)):
        try:
            module = importlib.import_module(f"pyod.models.{module_name}")
        except ImportError:  # no such module, or one needing an uninstalled library
            continue
        if isinstance(getattr(module, name, None), type):
            return model_class(module, name, PYOD_OPTION, base=base)
    return model_class(models, name, PYOD_OPTION, base=base)  # raises: not found


def _robust_scale(x: np.ndarray) -> np.ndarray:
    """Centre on the median and divide by a robust spread (MAD, else std, else 1)."""
    centre = np.median(x)
    spread = _MAD_TO_SIGMA * np.median(np.abs(x - centre))
    if not spread > 0:
        spread = float(np.std(x))
    if not spread > 0:
        spread = 1.0
    return (x - centre) / spread


def _noise_sigma(x: np.ndarray) -> float:
    """Estimate sensor noise from the first differences (MAD / sqrt 2).

    Differences cancel slow cycles and trends, so this is the sample-to-sample
    noise, not the signal's range. Returns 0 for a constant signal.
    """
    if len(x) < 3:
        return 0.0
    d = np.diff(x)
    sigma = _MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    if not sigma > 0:
        sigma = float(np.std(d)) / np.sqrt(2.0)
    return float(sigma)


def _check_positive(value: float, field: str, method: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise SignalConfigError(
            f"{method}: {field} must be a positive number, got {value!r}.",
            field=field,
            value=value,
        )


# -- pyod -----------------------------------------------------------------------------


def _pyod_features(present: pd.Series, features: list[str], window: str) -> np.ndarray:
    columns = []
    for name in features:
        if name == "value":
            columns.append(present.to_numpy(dtype="float64"))
        elif name == "diff":
            columns.append(present.diff().fillna(0.0).to_numpy(dtype="float64"))
        else:  # residual
            try:
                width = pd.Timedelta(window)
            except (TypeError, ValueError):
                raise SignalConfigError(
                    f"PyodDetector: window {window!r} isn't a duration, e.g. 1h.",
                    field="window",
                    value=window,
                ) from None
            if width <= pd.Timedelta(0):
                raise SignalConfigError(
                    f"PyodDetector: window must be positive, got {window!r}.",
                    field="window",
                    value=window,
                )
            median = present.rolling(width, center=True, min_periods=1).median()
            columns.append((present - median).to_numpy(dtype="float64"))
    return np.column_stack([_robust_scale(c) for c in columns])


def pyod_detector(
    values: pd.Series,
    *,
    model: ModelName = "IForest",
    contamination: float = 0.01,
    features: list[Literal["value", "diff", "residual"]] = _PYOD_FEATURES,
    window: str = "1h",
    seed: int = 0,
    params: dict[str, Any] = {},  # noqa: B006 - an option default, copied by pydantic
) -> np.ndarray:
    """Reject the values a pyod outlier detector scores as most unusual.

    The detector is trained on the stream it cleans, each time the step runs, on
    a few features built from each present value: the value itself, its change
    from the previous value, and its distance from the rolling median of its
    neighbours (the residual, which makes a spike stand out even on a stream
    with a strong daily cycle). Each feature is scaled robustly, so no single
    one dominates. The ``contamination`` share of values with the highest
    scores are rejected and flagged ``OUTLIER``; missing values are skipped.

    Use it when the bad values aren't simply out of range or too fast, and you'd
    rather let a well-tested detector find them than tune a threshold. Note that
    pyod always rejects about ``contamination`` of the values, so set it near
    the share of bad values you expect. For a stream with a known clean period,
    a fitted threshold method such as ``hampel`` may be the simpler choice.

    Args:
        values: The stream, NaN where missing.
        model: The pyod detector, by class name. Tested: ``IForest``
            (default; Isolation Forest, seeded by ``seed``), ``ECOD`` and
            ``COPOD`` (no parameters; they rank values by how far into the tails
            they are, so ties among extreme values are common), ``HBOS``
            (histograms; fast), ``KNN`` (distance to the 5th nearest value) and
            ``LOF`` (local density). Any other pyod detector, e.g. ``OCSVM`` or
            ``PCA``, runs too, untested; its own arguments go in ``params``.
        contamination: The share of present values to reject, above 0 and at
            most 0.5; e.g. 0.01 rejects the 1% most unusual values.
        features: What the detector sees for each value: ``value``, ``diff``
            (change from the previous present value) and ``residual`` (distance
            from the rolling median over ``window``). At least one.
        window: The centred rolling-median window for the ``residual``
            feature, a duration such as ``1h``.
        seed: The random seed for detectors that use one (``IForest``), so
            repeated runs give the same result.
        params: Other arguments of the detector, e.g. ``{"n_neighbors": 10}``
            for ``KNN`` or ``LOF``; checked against it when the step runs.

    Returns:
        True for each value to reject.
    """
    method = "PyodDetector"
    if not 0 < contamination <= 0.5:
        raise SignalConfigError(
            f"{method}: contamination must be above 0 and at most 0.5, got "
            f"{contamination!r}.",
            field="contamination",
            value=contamination,
        )
    if not features:
        raise SignalConfigError(
            f"{method}: features needs at least one of value, diff, residual.",
            field="features",
        )
    cls = _pyod_class(model, method)
    mask = np.zeros(len(values), dtype=bool)
    present_rows = values.notna().to_numpy()
    n = int(present_rows.sum())
    if n == 0:
        return mask
    if n < _PYOD_MIN_VALUES:
        raise SignalDataError(
            f"{method} on {values.name!r} has only {n} values to train on; it "
            f"needs at least {_PYOD_MIN_VALUES}."
        )
    X = _pyod_features(values[present_rows], list(dict.fromkeys(features)), window)
    settings = {"contamination": float(contamination), "random_state": int(seed)}
    detector = build_model(
        cls, settings, params, PYOD_OPTION, managed=frozenset({"contamination"})
    )
    detector.fit(X)
    mask[present_rows] = np.asarray(detector.labels_, dtype=bool)
    return mask


# -- ruptures -------------------------------------------------------------------------


def _segment_level(s: np.ndarray, start: int, stop: int) -> float:
    return float(np.median(s[start:stop]))


def ruptures_changepoint(
    values: pd.Series,
    *,
    algorithm: ModelName = "Pelt",
    cost: Literal["l2", "l1"] = "l2",
    penalty: float | None = None,
    max_segment: str = "30min",
    min_shift: float = 5.0,
    min_size: int = 1,
    params: dict[str, Any] = {},  # noqa: B006 - an option default, copied by pydantic
) -> np.ndarray:
    """Reject short segments that jump away from the level on both sides.

    ruptures splits the stream into segments of steady level at its change
    points. A segment no longer than ``max_segment`` whose level is above both
    neighbouring segments, or below both, by more than ``min_shift`` noise
    standard deviations is a transient glitch, e.g. a sensor that read high for
    20 minutes and came back: its values are rejected and flagged ``OUTLIER``.
    A run of consecutive short segments that together form one glitch is
    treated the same way. Long segments are regimes (a pump starting, a real
    step change) and are left alone, as are segments at the very start or end
    of the data, which have only one neighbour.

    The noise level is estimated from the sample-to-sample differences, so a
    slow daily cycle doesn't count as noise. Missing values are skipped: the
    change points are found on the present values. Needs a regular time index.

    Use it for level-shift glitches lasting several samples, which a spike
    filter such as ``hampel`` misses or only trims at the edges.

    Args:
        values: The stream, NaN where missing.
        algorithm: The ruptures search, by class name. Tested: ``Pelt``
            (default; exact), ``Binseg`` and ``BottomUp`` (approximate, faster
            on long streams). ``Window`` and ``KernelCPD`` run too, untested;
            their own arguments go in ``params``.
        cost: The segment cost: ``l2`` (default; change in mean) or ``l1``
            (change in median; more robust to spikes).
        penalty: The cost of adding a change point, on the stream scaled to
            unit noise. None (default) uses 2·log(n). Raise it to find fewer,
            larger changes.
        max_segment: The longest segment that can be rejected, a duration such
            as ``30min``. Longer segments are kept as real regimes.
        min_shift: How far, in noise standard deviations, a short segment's
            level must be from both neighbours to be rejected.
        min_size: The fewest samples in a segment. 1 (default) lets a single
            sample be its own segment.
        params: Other arguments of the ruptures algorithm, e.g.
            ``{"width": 20}`` for ``Window``.

    Returns:
        True for each value to reject.
    """
    method = "RupturesChangepoint"
    _check_positive(min_shift, "min_shift", method)
    if penalty is not None:
        _check_positive(penalty, "penalty", method)
    if min_size < 1:
        raise SignalConfigError(
            f"{method}: min_size must be at least 1, got {min_size!r}.",
            field="min_size",
            value=min_size,
        )
    step = regular_step(values.index, method)
    longest = steps_in(max_segment, step, field="max_segment", method=method)
    rpt = require("ruptures", EXTRA, method=method)

    mask = np.zeros(len(values), dtype=bool)
    present_rows = np.flatnonzero(values.notna().to_numpy())
    x = values.to_numpy(dtype="float64")[present_rows]
    n = len(x)
    if n < 3 * min_size or n < 3:
        return mask
    sigma = _noise_sigma(x)
    if not sigma > 0:
        return mask
    s = (x - np.median(x)) / sigma
    base = require("ruptures.base", EXTRA, method=method).BaseEstimator
    cls = model_class(rpt, algorithm, RUPTURES_OPTION, base=base)
    settings: dict[str, Any] = {"model": cost, "min_size": min_size, "jump": 1}
    takes = inspect.signature(cls.__init__).parameters
    if "model" not in takes and "kernel" in takes:  # KernelCPD
        if cost != "l2":
            raise SignalConfigError(
                f"{method}: ruptures {algorithm} takes a kernel, not a cost, so "
                f"cost={cost!r} can't be used; use cost='l2' (its default linear "
                'kernel) or give another kernel in params, e.g. {"kernel": "rbf"}.',
                field="cost",
                value=cost,
            )
        settings["kernel"] = "linear"
    algo = build_model(
        cls,
        settings,
        params,
        RUPTURES_OPTION,
        managed=frozenset({"model", "min_size", "jump"}),
    )
    try:
        breaks = algo.fit(s).predict(
            pen=2.0 * np.log(n) if penalty is None else penalty
        )
    except TypeError as exc:
        raise SignalConfigError(
            f"{method}: ruptures {algorithm} can't search with a penalty ({exc}); "
            f"use {', '.join(_RUPTURES_ALGORITHMS)}, Window or KernelCPD.",
            field="algorithm",
            value=algorithm,
        ) from None
    bounds = [0, *breaks]
    if bounds[-1] != n:  # pragma: no cover - ruptures always ends at n
        bounds.append(n)
    segments = list(zip(bounds[:-1], bounds[1:], strict=True))
    levels = [_segment_level(s, a, b) for a, b in segments]
    positions = values.index.asi8[present_rows]
    step_ns = step.value
    max_ns = longest * step_ns

    def duration(first: int, last: int) -> int:
        """Time covered by segments first..last, in ns."""
        return int(positions[segments[last][1] - 1] - positions[segments[first][0]])

    reject = np.zeros(n, dtype=bool)
    for first in range(1, len(segments) - 1):
        for last in range(first, len(segments) - 1):
            if duration(first, last) + step_ns > max_ns:
                break
            width = segments[last][1] - segments[first][0]
            left_a, left_b = segments[first - 1]
            right_a, right_b = segments[last + 1]
            reference = max(width, 3)
            left = _segment_level(s, max(left_a, left_b - reference), left_b)
            right = _segment_level(s, right_a, min(right_b, right_a + reference))
            inner = np.array(levels[first : last + 1])
            up = (inner - max(left, right) > min_shift).all()
            down = (min(left, right) - inner > min_shift).all()
            if up or down:
                reject[segments[first][0] : segments[last][1]] = True
    mask[present_rows] = reject
    return mask


# -- stumpy ---------------------------------------------------------------------------


def stumpy_discord(
    values: pd.Series,
    *,
    window: str = "1h",
    k: float = 6.0,
    normalize: bool = False,
) -> np.ndarray:
    """Reject the samples of stretches unlike anything else in the stream.

    stumpy's matrix profile gives, for every stretch of ``window`` length, the
    distance to its most similar stretch elsewhere in the stream. A stretch
    with no close match is a *discord*: a shape the stream never repeats. This
    step scores each stretch's distance by a robust z-score (its distance from
    the median distance, in median-absolute-deviation units) and calls the
    stretches above ``k`` anomalous. It rejects a sample when every stretch
    containing it is anomalous, so a spike rejects the spike, not the whole
    window around it. Rejected values are flagged ``OUTLIER``.

    Missing values are never rejected. For the comparison only, gaps are
    bridged by straight lines (stumpy would otherwise give every stretch that
    touches a gap an infinite distance); a stretch more than half missing is
    left out of the scores and never called anomalous. Needs a regular time
    index, and a stream that repeats, such as a daily
    cycle, over at least a few windows. The matrix profile finds *unique*
    anomalies: the same fault repeated many times matches itself and isn't a
    discord. The first run in a Python session compiles stumpy (tens of
    seconds).

    Use it for odd shapes and glitches on a periodic stream when you can't say
    in advance what a fault looks like.

    Args:
        values: The stream, NaN where missing.
        window: The stretch length, a duration such as ``1h``: about as long
            as the faults to find. At least 3 samples.
        k: The robust z-score above which a stretch is anomalous. Lower it to
            reject more.
        normalize: Compare stretches by shape only (z-normalized, stumpy's
            default) instead of by their actual values (False, the default
            here). Shape-only comparison misses level glitches and spikes,
            which look alike once normalized.

    Returns:
        True for each value to reject.
    """
    method = "StumpyDiscord"
    _check_positive(k, "k", method)
    step = regular_step(values.index, method)
    m = steps_in(window, step, field="window", method=method)
    if m < 3:
        raise SignalConfigError(
            f"{method}: window {window} is {m} samples; it must be at least 3.",
            field="window",
            value=window,
        )
    x = values.to_numpy(dtype="float64")
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    if n < 4 * m:
        raise SignalDataError(
            f"{method} on {values.name!r} needs at least 4 windows of data "
            f"({4 * m} samples); it has {n}. Use a shorter window."
        )
    present = np.isfinite(x)
    if not present.any():
        return mask
    stumpy = require("stumpy", EXTRA, method=method)
    # Bridge gaps with straight lines for the comparison only, so a stretch with
    # a few missing samples still gets a distance; a stretch more than half
    # missing is never scored or called anomalous.
    filled = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype="float64")
    profile = stumpy.stump(filled, m, normalize=normalize)
    distance = np.asarray(profile[:, 0], dtype="float64")
    missing = np.convolve((~present).astype(int), np.ones(m, dtype=int), "valid")
    distance[2 * missing > m] = np.inf
    finite = np.isfinite(distance)
    if finite.sum() < 3:
        return mask
    centre = np.median(distance[finite])
    spread = _MAD_TO_SIGMA * np.median(np.abs(distance[finite] - centre))
    if not spread > 0:
        return mask
    anomalous = finite & ((distance - centre) / spread > k)
    # A sample is rejected when every stretch containing it is anomalous.
    kernel = np.ones(m, dtype=int)
    covering = np.convolve(np.ones(len(distance), dtype=int), kernel)[:n]
    flagged = np.convolve(anomalous.astype(int), kernel)[:n]
    return (flagged > 0) & (flagged == covering) & present


# -- registration ---------------------------------------------------------------------

for _name, (_module, _description) in _PYOD_MODELS.items():
    register_backend(
        f"pyod:{_name}",
        f"{_module}:{_name}",
        extra=EXTRA,
        description=_description,
        docs_url="https://pyod.readthedocs.io/en/latest/pyod.models.html",
    )
for _name, _description in _RUPTURES_ALGORITHMS.items():
    register_backend(
        f"ruptures:{_name}",
        f"ruptures:{_name}",
        extra=EXTRA,
        description=f"Change-point detection: {_description[0].lower()}"
        f"{_description[1:]}",
        docs_url="https://centre-borelli.github.io/ruptures-docs/",
    )
register_backend(
    "stumpy:stump",
    "stumpy:stump",
    extra=EXTRA,
    description="Matrix profile: each stretch's distance to its nearest match.",
    docs_url="https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stump",
)

register_cleaner("PyodDetector", pyod_detector, flag="OUTLIER")
register_cleaner("RupturesChangepoint", ruptures_changepoint, flag="OUTLIER")
register_cleaner("StumpyDiscord", stumpy_discord, flag="OUTLIER")
