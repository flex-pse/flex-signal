"""Seeded, plant-like synthetic data with known injected faults.

This is flex-signal's main source of test, example and benchmark data (PLAN.md
§8, "Test data"). Every case keeps three things:

- the underlying noise-free signal (``underlying``);
- the fault-free measurement (``clean``: the signal plus sensor noise);
- the observed data with faults injected (``observed``), together with an exact
  boolean mask of where each fault was injected.

Because the ground truth is known, tests can assert that a cleaner flags
*exactly* the injected faults, and that an imputer recovers values to a known
accuracy.

Example:
    >>> from flexsignal.testing import synthetic
    >>> case = synthetic.plant(seed=1).with_spikes("flow", count=5)
    >>> int(case.fault_mask("spike", "flow").sum())
    5
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from flexsignal.core.frame import SignalFrame

__all__ = ["STREAMS", "SyntheticCase", "plant"]

STREAMS: dict[str, dict[str, object]] = {
    "flow": {"units": "m^3/hr", "noise": 1.0},
    "pressure": {"units": "bar", "noise": 0.02},
    "temperature": {"units": "degC", "noise": 0.1},
    "pump_on": {"units": None, "noise": 0.0},
    "reference_flow": {"units": "m^3/hr", "noise": 0.2},
}
"""Streams :func:`plant` generates, with units and sensor-noise standard deviation."""


@dataclass(frozen=True)
class SyntheticCase:
    """A synthetic dataset with known faults; ``with_*`` methods return a new case."""

    underlying: pd.DataFrame
    """The noise-free signal."""
    clean: pd.DataFrame
    """The fault-free measurement, i.e. the signal plus sensor noise."""
    observed: pd.DataFrame
    """The measurement with every injected fault applied."""
    units: dict[str, str | None]
    """Units per stream."""
    noise: dict[str, float]
    """Sensor-noise standard deviation per stream."""
    seed: int
    """Base seed. Each injection draws from its own stream derived from it."""
    faults: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Fault kind → boolean frame that is True where that fault was injected."""

    # -- access --------------------------------------------------------------------

    def signal(self) -> SignalFrame:
        """Return the observed data as a SignalFrame, with units."""
        return SignalFrame.from_frame(
            self.observed, units={k: v for k, v in self.units.items() if v}
        )

    def fault_mask(
        self, kind: str, stream: str | None = None
    ) -> pd.DataFrame | pd.Series:
        """Return where a fault kind was injected.

        Args:
            kind: e.g. ``"spike"``, ``"stuck"``, ``"gap"``, ``"out_of_range"``.
            stream: One stream, or None for all streams.

        Returns:
            A boolean Series (one stream) or DataFrame. All False if the fault was
            never injected.
        """
        mask = self.faults.get(kind)
        if mask is None:
            mask = pd.DataFrame(
                False, index=self.observed.index, columns=self.observed.columns
            )
        return mask[stream] if stream is not None else mask

    def any_fault(self, stream: str) -> pd.Series:
        """Return where any fault was injected into ``stream``."""
        out = pd.Series(False, index=self.observed.index)
        for mask in self.faults.values():
            out |= mask[stream]
        return out

    # -- injection helpers -----------------------------------------------------------

    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, len(self.faults), salt])

    def _free_starts(self, stream: str, length: int, margin: int) -> np.ndarray:
        """Start positions where a run of ``length`` fits clear of earlier faults."""
        busy = self.any_fault(stream).to_numpy()
        n = len(busy)
        padded = (
            np.convolve(
                busy.astype(int), np.ones(2 * margin + 1, dtype=int), mode="same"
            )
            > 0
        )
        starts = []
        for start in range(margin, n - length - margin):
            if not padded[start : start + length].any():
                starts.append(start)
        return np.array(starts, dtype=int)

    def _pick_runs(
        self, stream: str, count: int, length: int, spacing: int, salt: int
    ) -> list[int]:
        rng = self._rng(salt)
        chosen: list[int] = []
        candidates = self._free_starts(stream, length, spacing)
        rng.shuffle(candidates)
        for start in candidates:
            if all(abs(start - c) > length + spacing for c in chosen):
                chosen.append(int(start))
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            raise ValueError(
                f"only room for {len(chosen)} of {count} {length}-sample faults in "
                f"{stream!r} with spacing {spacing}; use fewer or a longer dataset."
            )
        return sorted(chosen)

    def _with_fault(
        self, kind: str, stream: str, positions: np.ndarray, new_values: np.ndarray
    ) -> SyntheticCase:
        observed = self.observed.copy()
        column = observed[stream].to_numpy(copy=True)
        column[positions] = new_values
        observed[stream] = column
        faults = dict(self.faults)
        mask = self.fault_mask(kind).copy()
        col_mask = mask[stream].to_numpy(copy=True)
        col_mask[positions] = True
        mask[stream] = col_mask
        faults[kind] = mask
        return replace(self, observed=observed, faults=faults)

    def _check_stream(self, stream: str) -> None:
        if stream not in self.observed.columns:
            raise ValueError(
                f"no stream {stream!r}; streams are {list(self.observed.columns)}"
            )

    # -- faults ------------------------------------------------------------------------

    def with_spikes(
        self, stream: str, *, count: int = 5, size: float = 40.0, spacing: int = 30
    ) -> SyntheticCase:
        """Add isolated one-sample spikes, ``size`` noise standard deviations high.

        Spikes alternate up and down, and are at least ``spacing`` samples from
        each other and from earlier faults. Fault kind: ``"spike"``.
        """
        self._check_stream(stream)
        starts = np.array(self._pick_runs(stream, count, 1, spacing, salt=1))
        sigma = self.noise[stream] or 1.0
        signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        values = self.observed[stream].to_numpy()[starts] + signs * size * sigma
        return self._with_fault("spike", stream, starts, values)

    def with_stuck(
        self, stream: str, *, count: int = 1, length: int = 36, spacing: int = 30
    ) -> SyntheticCase:
        """Freeze the sensor: hold the first value of a run for ``length`` samples.

        Fault kind: ``"stuck"``. The mask covers the whole run, including its
        first sample, which keeps its true value but is part of the flat line.
        """
        self._check_stream(stream)
        starts = self._pick_runs(stream, count, length, spacing, salt=2)
        case = self
        for start in starts:
            positions = np.arange(start, start + length)
            held = np.full(length, case.observed[stream].iloc[start])
            case = case._with_fault("stuck", stream, positions, held)
        return case

    def with_gaps(
        self, stream: str, *, count: int = 3, length: int = 2, spacing: int = 30
    ) -> SyntheticCase:
        """Remove ``count`` runs of ``length`` consecutive samples (set them to NaN).

        Fault kind: ``"gap"``.
        """
        self._check_stream(stream)
        starts = self._pick_runs(stream, count, length, spacing, salt=3)
        positions = np.concatenate([np.arange(s, s + length) for s in starts])
        return self._with_fault(
            "gap", stream, positions, np.full(len(positions), np.nan)
        )

    def with_out_of_range(
        self, stream: str, *, count: int = 5, value: float = -999.0, spacing: int = 30
    ) -> SyntheticCase:
        """Replace ``count`` samples with a sentinel such as an error code.

        Fault kind: ``"out_of_range"``.
        """
        self._check_stream(stream)
        starts = np.array(self._pick_runs(stream, count, 1, spacing, salt=4))
        return self._with_fault("out_of_range", stream, starts, np.full(count, value))

    def with_drift(
        self, stream: str, *, start: int, rate_per_day: float
    ) -> SyntheticCase:
        """Add a linear calibration drift from sample ``start`` onward.

        Fault kind: ``"drift"``. The mask covers every drifted sample.
        """
        self._check_stream(stream)
        index = self.observed.index
        elapsed_days = (
            index[start:] - index[start]
        ).total_seconds().to_numpy() / 86400.0
        positions = np.arange(start, len(index))
        values = self.observed[stream].to_numpy()[start:] + rate_per_day * elapsed_days
        return self._with_fault("drift", stream, positions, values)

    def with_step(self, stream: str, *, start: int, size: float) -> SyntheticCase:
        """Add a constant offset from sample ``start`` on, like a bad recalibration.

        Fault kind: ``"step"``.
        """
        self._check_stream(stream)
        positions = np.arange(start, len(self.observed))
        values = self.observed[stream].to_numpy()[start:] + size
        return self._with_fault("step", stream, positions, values)


def plant(
    *,
    start: str = "2025-01-06",
    periods: int = 7 * 288,
    freq: str = "5min",
    seed: int = 0,
    pump_off_periods: int = 0,
    reference_hours: tuple[int, int] = (9, 11),
    noise: Mapping[str, float] | None = None,
) -> SyntheticCase:
    """Generate a plant-like dataset with no faults yet.

    The streams:

    - ``flow`` (m^3/hr): around 100, with a daily cycle (±30, peaking in the
      afternoon) and a weekly cycle (±5). It is 0 whenever the pump is off.
    - ``pressure`` (bar): 4 + 0.01 × flow, so it correlates with flow.
    - ``temperature`` (degC): 15, with a daily cycle of ±3.
    - ``pump_on`` (0/1): 1, except in ``pump_off_periods`` random blocks of 2–6 h.
    - ``reference_flow`` (m^3/hr): a trusted, precise meter for the same flow. It
      reports only between ``reference_hours`` each day, and is NaN otherwise.
      M2's ``calibrate`` uses it.

    Args:
        start: First timestamp (UTC).
        periods: Number of samples.
        freq: Sample spacing.
        seed: Seed for the noise, pump-off blocks and every later fault injection.
        pump_off_periods: How many pump-off blocks to add.
        reference_hours: ``(first_hour, end_hour)``, in UTC, of the reference
            meter's daily reporting window.
        noise: Override the sensor-noise standard deviation per stream.

    Returns:
        A :class:`SyntheticCase` with ``observed == clean``.
    """
    rng = np.random.default_rng([seed, 0])
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="time")
    hours = (index - index[0]).total_seconds().to_numpy() / 3600.0
    hour_of_day = index.hour.to_numpy() + index.minute.to_numpy() / 60.0

    pump_on = np.ones(periods)
    step_hours = pd.Timedelta(freq).total_seconds() / 3600.0
    for _ in range(pump_off_periods):
        length = int(rng.uniform(2, 6) / step_hours)
        begin = int(rng.integers(0, max(periods - length, 1)))
        pump_on[begin : begin + length] = 0.0

    daily = np.sin(2 * np.pi * (hour_of_day - 9.0) / 24.0)
    weekly = np.sin(2 * np.pi * hours / (24.0 * 7))
    flow = (100.0 + 30.0 * daily + 5.0 * weekly) * pump_on
    underlying = pd.DataFrame(
        {
            "flow": flow,
            "pressure": 4.0 + 0.01 * flow,
            "temperature": 15.0 + 3.0 * daily,
            "pump_on": pump_on,
            "reference_flow": flow,
        },
        index=index,
    )
    sigma = {name: float(info["noise"]) for name, info in STREAMS.items()}
    sigma.update(noise or {})
    clean = underlying.copy()
    for name in clean.columns:
        if sigma[name]:
            clean[name] = clean[name] + rng.normal(0.0, sigma[name], periods)
    in_window = (hour_of_day >= reference_hours[0]) & (hour_of_day < reference_hours[1])
    clean.loc[~in_window, "reference_flow"] = np.nan
    underlying.loc[~in_window, "reference_flow"] = np.nan
    units = {name: info["units"] for name, info in STREAMS.items()}
    return SyntheticCase(
        underlying=underlying,
        clean=clean,
        observed=clean.copy(),
        units=units,  # type: ignore[arg-type]
        noise=sigma,
        seed=seed,
    )
