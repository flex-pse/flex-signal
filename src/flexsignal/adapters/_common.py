"""Helpers the adapters share: optional libraries, model names, grids and seeds."""

from __future__ import annotations

import importlib
import inspect
import random
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import Field

from flexsignal.core.exceptions import (
    SignalBackendError,
    SignalConfigError,
    SignalDataError,
)


def require(module: str, extra: str, *, method: str) -> ModuleType:
    """Import an optional library, or say which extra installs it.

    Args:
        module: The module to import, e.g. ``"pyod.models.ecod"``.
        extra: The flex-signal extra that installs it, e.g. ``"anomaly"``.
        method: The registered method that needs it, for the message.

    Returns:
        The imported module.

    Raises:
        SignalBackendError: If the library isn't installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SignalBackendError(
            f"{method} needs {module.split('.')[0]!r}, which is not installed: "
            f"install it with `pip install flex-signal[{extra}]`.",
            value=method,
        ) from exc


def regular_step(index: pd.DatetimeIndex, method: str) -> pd.Timedelta:
    """Return the single spacing of a regular index, or raise naming the method."""
    if len(index) < 2:
        raise SignalDataError(f"{method} needs at least two timestamps.")
    diffs = np.unique(np.diff(index.asi8))
    if len(diffs) != 1:
        raise SignalDataError(
            f"{method} needs a regular time index; put an align or resample step "
            "before it."
        )
    return pd.Timedelta(int(diffs[0]), unit="ns")


def steps_in(duration: str, step: pd.Timedelta, *, field: str, method: str) -> int:
    """Return how many data steps a duration option covers (at least 1)."""
    try:
        delta = pd.Timedelta(duration)
    except (TypeError, ValueError):
        raise SignalDataError(
            f"{method}: {field} {duration!r} isn't a duration, e.g. 1h."
        ) from None
    n, remainder = divmod(delta, step)
    if remainder != pd.Timedelta(0) or n < 1:
        raise SignalDataError(
            f"{method}: {field} {duration} isn't a whole number of the data's {step} "
            "steps."
        )
    return int(n)


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed Python, numpy and torch (if installed) so a run repeats, for a block.

    Inside the block the random generators start from ``seed`` and torch uses
    deterministic algorithms. Afterwards the generators' states and torch's
    deterministic setting are put back, so the caller's own random numbers
    don't change.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
        except ImportError:  # pragma: no cover - torch-free adapters
            yield
            return
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                yield
            finally:
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@contextmanager
def quiet_pandas(*filters: tuple[str, type[Warning]]) -> Iterator[None]:
    """Hide pandas' deprecation warnings from inside a library, and ``filters``.

    A user can't act on a library's own deprecation warnings. Each filter is a
    ``(message regex, warning category)`` pair to hide as well.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
        for message, category in filters:
            warnings.filterwarnings("ignore", message=message, category=category)
        yield


# -- forecasts ------------------------------------------------------------------------

_QUANTILE_KEY = re.compile(r"q(\d\d)")


def quantile_key(q: float) -> str:
    """Return the key a forecast gives quantile ``q`` under, e.g. ``q10`` for 0.1."""
    return f"q{round(q * 100):02d}"


class Forecast(Mapping):
    """A forecast's mean, and the quantiles its model can give.

    flex-signal looks up ``"q10"``-style keys only when a step has
    ``quantiles``. ``quantiles`` is either a function from a quantile (e.g. 0.1)
    to its values, computed when first read, so any whole-percent quantile can
    be read; a dict of the quantiles the model gave, by key; or None when the
    model gives none. Looking up a quantile the model can't give raises a
    :class:`SignalConfigError` that says why (``refusal``: a message, or a
    function from the requested quantile to one), instead of a generic error.
    """

    def __init__(
        self,
        mean: np.ndarray,
        quantiles: Callable[[float], np.ndarray] | Mapping[str, np.ndarray] | None,
        refusal: str | Callable[[float], str],
    ):
        self._mean = mean
        self._quantiles = quantiles
        self._refusal = refusal
        self._cache: dict[str, np.ndarray] = {}

    def _gives(self, key: str) -> bool:
        if self._quantiles is None:
            return False
        return callable(self._quantiles) or key in self._quantiles

    def __contains__(self, key: object) -> bool:
        if key == "mean":
            return True
        if not isinstance(key, str) or not _QUANTILE_KEY.fullmatch(key):
            return False
        if self._gives(key):
            return True
        q = int(key[1:]) / 100
        message = self._refusal(q) if callable(self._refusal) else self._refusal
        raise SignalConfigError(message, field="quantiles", value=q)

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "mean":
            return self._mean
        if not isinstance(key, str) or not _QUANTILE_KEY.fullmatch(key):
            raise KeyError(key)
        if not callable(self._quantiles):
            if self._quantiles is None:
                raise KeyError(key)
            return self._quantiles[key]
        if key not in self._cache:
            self._cache[key] = self._quantiles(int(key[1:]) / 100)
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        if self._quantiles is None or callable(self._quantiles):
            return iter(("mean",))
        return iter(("mean", *self._quantiles))

    def __len__(self) -> int:
        return sum(1 for _ in self)


# -- the history as a statistics library wants it -------------------------------------


def naive_index(index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DatetimeIndex:
    """The (UTC) index as naive UTC times carrying the data's frequency."""
    naive = index.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(naive, freq=pd.tseries.frequencies.to_offset(step))


class Prepared:
    """A forecast step's target history and exog streams, ready for a library.

    The target runs from its first to its last value, gaps interpolated; the
    exog streams cover the same start to the forecast's end, gaps interpolated.
    """

    def __init__(self, history: pd.DataFrame, future: pd.DataFrame, method: str):
        self.stream = str(history.columns[0])
        self.step = regular_step(history.index, method)
        target = history.iloc[:, 0].astype("float64")
        valid = np.flatnonzero(target.notna().to_numpy())
        if len(valid) < 2:
            raise SignalDataError(
                f"{method}: {self.stream!r} has fewer than two values up to the "
                "issue time, so there is nothing to fit on."
            )
        first, last = int(valid[0]), int(valid[-1])
        y = target.iloc[first : last + 1].interpolate(method="linear")
        self.y = pd.Series(y.to_numpy(), index=naive_index(y.index, self.step))
        """The target from its first to its last value, gaps interpolated."""
        self.n_ahead = len(target) - 1 - last + len(future)
        """How many steps after the last value the forecast reaches (missing
        values at the end of the history are forecast along with the rows)."""
        self.future_index = naive_index(future.index, self.step)
        self.exog = list(history.columns[1:])
        self.covariates: pd.DataFrame | None = None
        """The exog streams from the target's first value to the forecast's end."""
        if self.exog:
            both = pd.concat([history[self.exog].iloc[first:], future[self.exog]])
            both = both.astype("float64").interpolate(limit_direction="both")
            empty = [c for c in self.exog if both[c].isna().all()]
            if empty:
                raise SignalDataError(
                    f"{method}: exog stream {empty[0]!r} has no values to use."
                )
            self.covariates = pd.DataFrame(
                both.to_numpy(),
                index=naive_index(both.index, self.step),
                columns=self.exog,
            )

    def check_times(self, times: pd.Index, method: str) -> None:
        """Check a library's forecast times are the forecast rows."""
        got = pd.DatetimeIndex(times[-len(self.future_index) :]).as_unit("ns")
        if len(times) != self.n_ahead or not np.array_equal(
            got.asi8, self.future_index.as_unit("ns").asi8
        ):
            raise SignalDataError(  # pragma: no cover - a library bug
                f"{method}: the library forecast times {got[0]}..{got[-1]}, not the "
                f"forecast rows {self.future_index[0]}..{self.future_index[-1]}."
            )


def season_steps(
    season: str | None, step: pd.Timedelta, method: str, *, used: bool = True
) -> int | None:
    """How many data steps the ``season`` option covers, or None without one.

    The option has a default (1D) that a user may never have set, so for a
    model that doesn't use it (``used=False``) it is only checked to be a
    duration, and None is returned.

    Raises:
        SignalConfigError: If ``season`` isn't a duration.
        SignalDataError: If it isn't a whole number of the data's steps.
    """
    if season is None:
        return None
    try:
        delta = pd.Timedelta(season)
    except (TypeError, ValueError):
        raise SignalConfigError(
            f"{method}: season {season!r} isn't a duration, e.g. 1D, or null.",
            field="season",
            value=season,
        ) from None
    if not used:
        return None
    n, remainder = divmod(delta, step)
    if remainder != pd.Timedelta(0) or n < 1:
        raise SignalDataError(
            f"{method}: season (default 1D) is {season}, which isn't a whole number "
            f"of the data's {step} steps; set season to a multiple of {step}, or "
            "set season=null if the signal has no cycle.",
            field="season",
            value=season,
        )
    return int(n)


# -- model names ----------------------------------------------------------------------

ModelName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
"""A library's model class, by bare name (e.g. ``SAITS``): never an import path."""


@dataclass(frozen=True)
class ModelOption:
    """An adapter option that names one of a library's models.

    The ``tested`` models run with the adapter's own simple settings; any other
    model class of the library runs too, untested, with its own settings in
    the step's ``params``. The methods reference page lists these.
    """

    method: str
    option: str
    library: str
    tested: tuple[str, ...]
    catalogue: str
    note: str = ""


MODEL_OPTIONS: list[ModelOption] = []
"""Every adapter's model option, in registration order."""


def model_option(**kwargs: Any) -> ModelOption:
    """Record an adapter's model option for the docs, and return it."""
    option = ModelOption(**kwargs)
    MODEL_OPTIONS[:] = [o for o in MODEL_OPTIONS if o.method != option.method]
    MODEL_OPTIONS.append(option)
    return option


def model_class(
    module: Any,
    name: str,
    option: ModelOption,
    *,
    base: type | tuple[type, ...] | None = None,
) -> type:
    """Return the model class ``name`` from a library module.

    Only public, concrete classes of that module (subclasses of ``base``, or of
    one of them, if given, but not the bases themselves) can be returned, so a
    spec can reach nothing but the library's models.

    Raises:
        SignalConfigError: If the library has no such model.
    """
    bases = () if base is None else base if isinstance(base, tuple) else (base,)
    found = None if name.startswith("_") else getattr(module, name, None)
    if (
        not isinstance(found, type)
        or inspect.isabstract(found)
        or found in bases
        or (bases and not issubclass(found, bases))
    ):
        raise SignalConfigError(
            f"{option.method}: {option.library} has no model {name!r}. Tested: "
            f"{', '.join(option.tested)}; any other {option.library} model works "
            f"too, by its class name: see {option.catalogue}.",
            field=option.option,
            value=name,
        )
    return found


_SET_BY_FLEX_SIGNAL = "flex-signal sets those itself, so use the step's own options"


def build_model(
    cls: type,
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
    option: ModelOption,
    *,
    managed: Iterable[str] = (),
    why_managed: str = _SET_BY_FLEX_SIGNAL,
    extra_accepted: Iterable[str] = (),
    strict: bool = False,
) -> Any:
    """Make a model from the adapter's ``settings`` and the user's ``params``.

    ``params`` may set any constructor argument except the ``managed`` ones,
    which the adapter sets itself (or a spec can't set, ``why_managed`` says
    why), and overrides ``settings``. A required argument that neither gives is
    named in the error, so an untested model says what it needs.

    Args:
        cls: The model class.
        settings: The adapter's own arguments. Those the class doesn't take are
            dropped, unless it takes ``**kwargs``.
        params: The step's ``params``.
        option: The adapter's model option, for the messages.
        managed: Arguments ``params`` can't set.
        why_managed: Why, for the message.
        extra_accepted: Arguments the class takes through ``**kwargs`` and
            passes on to the library it wraps.
        strict: Refuse ``params`` the class would only take through
            ``**kwargs`` (other than ``extra_accepted``).

    Raises:
        SignalConfigError: If ``params`` isn't a dict, names an unknown or
            managed argument, a required argument is missing, or the library
            refuses the settings.
    """
    if not isinstance(params, Mapping):
        raise SignalConfigError(
            f"{option.method}: params must be a dict of {option.library} "
            f"{cls.__name__} arguments, not {params!r}.",
            field="params",
            value=params,
        )
    managed = frozenset(managed)
    signature = inspect.signature(cls.__init__)
    accepted = {
        name
        for name, p in signature.parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    } | set(extra_accepted)
    takes_any = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    blocked = sorted(set(params) & managed)
    unknown = (
        [] if takes_any and not strict else sorted(set(params) - accepted - managed)
    )
    if blocked or unknown:
        parts = []
        if blocked:
            parts.append(f"params can't set {_names(blocked)}: {why_managed}")
        if unknown:
            are = "isn't an argument" if len(unknown) == 1 else "aren't arguments"
            parts.append(
                f"params {_names(unknown)} {are} of {option.library} {cls.__name__}"
            )
        raise SignalConfigError(
            f"{option.method}: {'; '.join(parts)}. {cls.__name__} accepts "
            f"{', '.join(sorted(accepted - managed))}.",
            field="params",
            value=blocked + unknown,
        )
    # The adapter's settings fit its tested models; another model takes the
    # ones its constructor has.
    kwargs = {k: v for k, v in settings.items() if takes_any or k in accepted}
    kwargs.update(params)
    missing = [
        name
        for name, p in signature.parameters.items()
        if name != "self"
        and p.default is p.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        and name not in kwargs
    ]
    if missing:
        tested = cls.__name__ in option.tested
        raise SignalConfigError(
            f"{option.method}: {option.library} {cls.__name__} needs {missing}"
            + (
                ""
                if tested
                else f", which flex-signal doesn't set for untested models: give "
                f"them in params, e.g. params={{{missing[0]!r}: ...}}. See "
                f"{option.catalogue}"
            )
            + ".",
            field="params",
            value=missing,
        )
    try:
        return cls(**kwargs)
    except (TypeError, ValueError, AssertionError) as exc:
        raise SignalConfigError(
            f"{option.method}: {option.library} {cls.__name__} refused these "
            f"settings: {exc}",
            field="params",
        ) from None


def _names(names: list[str]) -> str:
    return repr(names[0]) if len(names) == 1 else str(names)
