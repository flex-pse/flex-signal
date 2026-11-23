"""A masking benchmark for imputers: hide known values, fill them, score the fill.

Real data has no ground truth for its gaps. So the benchmark makes gaps where
the values *are* known: it hides runs of valid samples, runs each imputer on the
data with those values hidden, and compares what the imputer put back with what
was there. Repeating this for several gap lengths shows which method suits which
gaps, which is what a ``tiered`` imputer needs to know.

Gap lengths are measured the way imputers measure them, between the valid
samples either side. On 5-minute data a ``"15min"`` gap hides 2 samples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import FLAG_DTYPE, Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.eval.scores import metrics
from flexsignal.pipeline import Pipeline
from flexsignal.spec import StepSpec, to_timedelta
from flexsignal.steps._base import regular_step

__all__ = ["benchmark_imputers", "hide_values", "summarize"]

Imputer = StepSpec | Sequence[StepSpec]
"""An imputer to benchmark: one step, or several run in order."""


def hide_values(
    sf: SignalFrame,
    streams: Sequence[str],
    *,
    gap: str,
    count: int,
    pattern: Literal["block", "random"] = "block",
    seed: int = 0,
) -> tuple[SignalFrame, pd.DataFrame]:
    """Hide valid values so an imputer's fill can be checked against them.

    Args:
        sf: Data on a regular time grid.
        streams: Streams to hide values in.
        gap: Length of each gap, measured between the valid samples either side,
            e.g. ``"1h"``. It must be a whole number of steps, at least two.
        count: How many gaps to make in each stream.
        pattern: ``block``: each gap is one run of hidden samples. ``random``:
            the same number of samples, hidden one at a time at random places.
            No two of them are next to each other, so each is a one-sample gap
            between two valid samples.
        seed: Seed for where the gaps go.

    Returns:
        ``(masked, hidden)``: the data with the values hidden (NaN, flagged
        ``MISSING``), and a boolean frame that is True where a value was hidden.

    Raises:
        SignalConfigError: If ``gap`` isn't a whole number of steps (at least
            two), or ``count`` is less than 1.
        SignalDataError: If there isn't room for ``count`` gaps of this length.
    """
    step = regular_step(sf.index, kind="hide_values")
    periods, remainder = divmod(to_timedelta(gap).value, step.value)
    if remainder or periods < 2:
        raise SignalConfigError(
            f"gap {gap} must be a whole number of the data's {step} steps, and at "
            "least two of them.",
            field="gap",
        )
    if count < 1:
        raise SignalConfigError(
            f"count is {count}, but it must be at least 1: it is how many gaps to "
            "make in each stream.",
            field="count",
        )
    length = int(periods) - 1
    rng = np.random.default_rng(seed)
    values, flags = sf.values, sf.flags
    hidden = pd.DataFrame(False, index=sf.index, columns=list(streams))
    for col in streams:
        valid = values[col].notna().to_numpy()
        busy = ~valid
        positions = []
        if pattern == "block":
            # A gap and the valid sample on each side of it must all be free.
            window = np.convolve(busy.astype(int), np.ones(length + 2, int), "valid")
            starts = np.flatnonzero(window == 0) + 1
            rng.shuffle(starts)
            taken = np.zeros(len(valid), dtype=bool)
            for start in starts:
                if not taken[start - 1 : start + length + 1].any():
                    taken[start - 1 : start + length + 1] = True
                    positions.extend(range(start, start + length))
                    if len(positions) == count * length:
                        break
            if len(positions) < count * length:
                raise SignalDataError(
                    f"room for only {len(positions) // length} of {count} {gap} gaps "
                    f"in {col!r}; ask for fewer or shorter gaps."
                )
        else:
            interior = np.flatnonzero(valid[1:-1]) + 1
            neighbours = valid[interior - 1] & valid[interior + 1]
            candidates = interior[neighbours]
            rng.shuffle(candidates)
            # Each hidden sample keeps both neighbours valid, so no two picks
            # touch and every gap is a single sample.
            taken = np.zeros(len(valid), dtype=bool)
            for pick in candidates:
                if not taken[pick]:
                    taken[pick - 1 : pick + 2] = True
                    positions.append(pick)
                    if len(positions) == count * length:
                        break
            if len(positions) < count * length:
                raise SignalDataError(
                    f"room for only {len(positions)} of {count * length} separate "
                    f"single-sample gaps in {col!r}; ask for fewer or shorter gaps."
                )
        mask = np.zeros(len(valid), dtype=bool)
        mask[positions] = True
        hidden[col] = mask
        col_values = values[col].to_numpy(copy=True)
        col_values[mask] = np.nan
        values[col] = col_values
        col_flags = flags[col].to_numpy(copy=True)
        col_flags[mask] = FLAG_DTYPE(int(Flag.MISSING))
        flags[col] = col_flags
    return sf.evolve(values=values, flags=flags), hidden


def benchmark_imputers(
    sf: SignalFrame,
    imputers: Mapping[str, Imputer],
    *,
    streams: Sequence[str],
    gaps: Sequence[str] = ("15min", "1h", "6h"),
    count: int = 10,
    pattern: Literal["block", "random"] = "block",
    repeats: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Score imputers on values hidden from them.

    For each gap length and repeat, :func:`hide_values` hides ``count`` gaps in
    each stream. Every imputer is then fitted on, and applied to, the data with
    those values hidden, so it never sees them.

    Args:
        sf: Data on a regular time grid, ideally already cleaned.
        imputers: Name → imputer step (or steps), e.g.
            ``{"linear": Interpolate("flow", max_gap="1D")}``.
        streams: Streams to hide values in and score.
        gaps: Gap lengths to test.
        count: Gaps per stream, per gap length and repeat.
        pattern: ``block`` or ``random`` (see :func:`hide_values`).
        repeats: How many times to repeat with new gap positions.
        seed: Base seed.

    Returns:
        One row per imputer, stream, gap length and repeat, with columns
        ``imputer, stream, gap, repeat, hidden, filled, coverage, rmse, mae,
        bias``. ``coverage`` is the fraction of hidden values the imputer filled;
        the errors are over the values it filled.
    """
    if not imputers:
        raise SignalConfigError("give at least one imputer to benchmark.")
    original = sf.values
    rows = []
    for g_i, gap in enumerate(gaps):
        for repeat in range(repeats):
            masked, hidden = hide_values(
                sf,
                streams,
                gap=gap,
                count=count,
                pattern=pattern,
                seed=seed + 1000 * g_i + repeat,
            )
            for name, imputer in imputers.items():
                steps = [imputer] if isinstance(imputer, StepSpec) else list(imputer)
                _, filled_sf = Pipeline(steps, name=name).fit_transform(masked)
                filled = filled_sf.values
                for col in streams:
                    where = hidden[col].to_numpy()
                    truth = original[col].to_numpy()[where]
                    guess = filled[col].to_numpy()[where]
                    got = ~np.isnan(guess)
                    error = guess[got] - truth[got]
                    scores = metrics(pd.Series(truth), pd.Series(guess))
                    rows.append(
                        {
                            "imputer": name,
                            "stream": col,
                            "gap": gap,
                            "repeat": repeat,
                            "hidden": int(where.sum()),
                            "filled": int(got.sum()),
                            "coverage": float(got.mean()) if len(got) else np.nan,
                            "rmse": scores["rmse"],
                            "mae": scores["mae"],
                            "bias": float(np.mean(error)) if len(error) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Pivot benchmark results to one row per imputer and one column per gap.

    Args:
        results: What :func:`benchmark_imputers` returned.
        metric: The column to show, e.g. ``"rmse"`` or ``"coverage"``.

    Returns:
        The mean of ``metric`` over streams and repeats. Gaps keep the order
        they were tested in; imputers keep the order they were given in.
    """
    table = results.pivot_table(
        index="imputer", columns="gap", values=metric, aggfunc="mean", sort=False
    )
    return table[list(dict.fromkeys(results["gap"]))]
