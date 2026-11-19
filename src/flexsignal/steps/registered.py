"""Your own methods as steps: register a plain function, then use it by name.

Only registered methods can run. A method is registered by code,
in the user's script or an installed package, and a spec refers to it by name.
A spec never names an import path, and loading or running one never imports
anything. So a spec from someone else can run only what the person running it
chose to make available.

One function per role turns a plain Python function into a full step:

- :func:`~flexsignal.register_cleaner`: the function says which values to reject;
  flex-signal sets them to NaN, adds the flag, and skips values already missing.
- :func:`~flexsignal.register_imputer`: the function proposes values for the gaps it is
  given; flex-signal finds the gaps (``max_gap``, never ``EXCLUDED`` or
  ``NO_SAMPLE``), writes only into them, and flags them ``IMPUTED_MODEL``.
- :func:`~flexsignal.register_forecaster`: the function returns values for the horizon;
  flex-signal handles the issue time, the ``FORECAST`` flag, quantile streams
  and backtesting.
- :func:`~flexsignal.register_model`: a regressor becomes a
  ``model`` choice for ``calibrate`` and ``regression``.

The function's keyword-only parameters, with their type hints and defaults,
become the step's validated options. ``fit=`` optionally learns a JSON-able state
from training data.
"""

from __future__ import annotations

import inspect
import json
import keyword
import re
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from types import ModuleType
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field, create_model

from flexsignal.backends.models import registered_models
from flexsignal.core.exceptions import SignalConfigError, SignalDataError
from flexsignal.core.flags import Flag
from flexsignal.core.frame import SignalFrame
from flexsignal.core.registry import (
    find_step,
    get_step_class,
    load_entry_points,
    register_step,
    registered_steps,
)
from flexsignal.steps import clean, forecast, impute
from flexsignal.steps._base import StreamStep, check_streams, remove_values
from flexsignal.steps.forecast import ForecastStep, _Window, quantile_column
from flexsignal.steps.impute import _fill, _GapImputer

__all__ = [
    "Method",
    "load_plugins",
    "register_cleaner",
    "register_forecaster",
    "register_imputer",
    "registered_methods",
]

Role = Literal["cleaner", "imputer", "forecaster"]

CLEANER_FLAGS = (
    Flag.OUT_OF_RANGE,
    Flag.RATE_OF_CHANGE,
    Flag.OUTLIER,
    Flag.STUCK,
    Flag.MANUAL,
    Flag.EXCLUDED,
)
"""The flags a registered cleaner may set on the values it rejects."""

_DATA_ARGS: dict[str, tuple[str, ...]] = {
    "cleaner": ("values",),
    "imputer": ("values", "gaps"),
    "forecaster": ("history", "future"),
}
_MODULES: dict[str, ModuleType] = {
    "cleaner": clean,
    "imputer": impute,
    "forecaster": forecast,
}
_RESERVED: dict[str, frozenset[str]] = {
    "cleaner": frozenset({"kind", "name", "streams", "registered"}),
    "imputer": frozenset({"kind", "name", "streams", "max_gap", "registered"}),
    "forecaster": frozenset(
        {
            "kind",
            "name",
            "streams",
            "horizon",
            "issue_time",
            "exog",
            "quantiles",
            "registered",
        }
    ),
}


@dataclass(frozen=True)
class Method:
    """A registered method: the function behind a registered step."""

    role: str
    """``cleaner``, ``imputer`` or ``forecaster``."""
    name: str
    """The step's kind, and its name in fs.clean, fs.impute or fs.forecast."""
    function: Callable[..., Any]
    """The user's function."""
    fit: Callable[..., Any] | None
    """The optional fit function."""
    takes_state: bool
    """Whether the function takes the fitted state after its data arguments."""
    options: tuple[str, ...]
    """The option names, from the function's keyword-only parameters."""
    fit_options: tuple[str, ...] | None
    """The options ``fit`` accepts; None means all of them."""
    source: dict[str, Any] = field(default_factory=dict)
    """Where the function comes from (its module, name, package and version)."""
    flag: Flag | None = None
    """The flag a cleaner sets (None for other roles)."""


# -- names, signatures and provenance -------------------------------------------------


def _check_name(name: Any, role: str, source: Mapping[str, Any]) -> bool:
    """Check a registered name; return True if it replaces the same function."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise SignalConfigError(
            f"a registered {role} needs a name that is a Python identifier, e.g. "
            f"PlantDespike; got {name!r}.",
            value=name,
        )
    module = _MODULES[role]
    if name in vars(module):
        raise SignalConfigError(
            f"{name!r} is already fs.{module.__name__.rsplit('.', 1)[1]}.{name}, a "
            "built-in step; pick another name. Prefix names shared between teams, "
            "e.g. PlantDespike.",
            value=name,
        )
    existing = find_step(name)
    if existing is None:
        return False
    method = getattr(existing, "registered", None)
    same = method is not None and (
        method.source.get("module"),
        method.source.get("function"),
    ) == (source.get("module"), source.get("function"))
    if not same:
        owner = (
            f"{method.source.get('module')}.{method.source.get('function')}"
            if method is not None
            else f"the built-in step {existing.__module__}.{existing.__qualname__}"
        )
        raise SignalConfigError(
            f"step kind {name!r} is already registered by {owner}; pick another name.",
            value=name,
        )
    if method.role != role:
        raise SignalConfigError(
            f"{name!r} is already registered as a {method.role}; pick another name.",
            value=name,
        )
    return True


@lru_cache(maxsize=1)
def _distributions() -> Mapping[str, list[str]]:
    return packages_distributions()


def _source(fn: Callable[..., Any]) -> dict[str, Any]:
    module = getattr(fn, "__module__", None) or "<unknown>"
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", repr(fn))
    package = None
    package_version = None
    names = _distributions().get(module.split(".")[0]) or []
    if names:
        package = names[0]
        try:
            package_version = version(package)
        except PackageNotFoundError:  # pragma: no cover - a broken install
            package_version = None
    return {
        "module": module,
        "function": qualname,
        "package": package,
        "version": package_version,
    }


_ARG_LINE = re.compile(r"^(\s+)(\*{0,2}\w+)(?:\s*\([^)]*\))?:\s*(.*)$")


def _arg_docs(fn: Callable[..., Any]) -> dict[str, str]:
    """Read the ``Args:`` section of a Google-style docstring."""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    out: dict[str, str] = {}
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Args:")
    except StopIteration:
        return out
    current = None
    indent = None
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" "):
            break
        match = _ARG_LINE.match(line)
        if match and (indent is None or len(match.group(1)) == indent):
            indent = len(match.group(1))
            current = match.group(2).lstrip("*")
            out[current] = match.group(3).strip()
        elif current is not None and line.strip():
            out[current] = f"{out[current]} {line.strip()}"
    return out


def _summary(fn: Callable[..., Any], role: str, name: str) -> str:
    doc = inspect.getdoc(fn) or ""
    first = doc.split("\n\n", 1)[0].strip()
    source = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
    if first:
        return f"{first}\n\nA registered {role}: ``{source}``."
    return f"A registered {role}: ``{source}``."


def _signature(
    fn: Callable[..., Any], role: str, name: str
) -> tuple[bool, dict[str, Any]]:
    """Return (takes_state, pydantic fields) from the function's signature."""
    if not callable(fn):
        raise SignalConfigError(f"{name}: {fn!r} is not a function.", value=name)
    try:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn, include_extras=True)
    except (TypeError, ValueError, NameError) as exc:
        raise SignalConfigError(
            f"{name}: can't read the function's signature and type hints: {exc}.",
            value=name,
        ) from None
    data_args = _DATA_ARGS[role]
    usage = (
        f"a {role} is called as fn({', '.join(data_args)}[, state], *, option=...), "
        f"e.g. def {getattr(fn, '__name__', 'fn')}({', '.join(data_args)}, *, "
        "window: str = '1h')"
    )
    positional = []
    fields: dict[str, Any] = {}
    docs = _arg_docs(fn)
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            positional.append(p.name)
        elif p.kind is p.KEYWORD_ONLY:
            if p.name in _RESERVED[role]:
                raise SignalConfigError(
                    f"{name}: option {p.name!r} is set by flex-signal for every "
                    f"{role}; rename it.",
                    field=p.name,
                )
            if p.name not in hints:
                raise SignalConfigError(
                    f"{name}: give option {p.name!r} a type hint, e.g. "
                    f"{p.name}: float = 1.0, so its values can be checked.",
                    field=p.name,
                )
            description = docs.get(p.name) or f"Option {p.name} of {name}."
            if p.default is p.empty:
                fields[p.name] = (hints[p.name], Field(description=description))
            else:
                try:
                    json.dumps(p.default)
                except (TypeError, ValueError):
                    raise SignalConfigError(
                        f"{name}: option {p.name!r} has a default that isn't JSON "
                        f"({p.default!r}); use a number, string, bool, list, dict "
                        "or None.",
                        field=p.name,
                    ) from None
                fields[p.name] = (
                    hints[p.name],
                    Field(p.default, description=description),
                )
        else:
            raise SignalConfigError(
                f"{name}: the function takes *{p.name}; {usage}. Options must be "
                "keyword-only parameters, after the *.",
                value=name,
            )
    if len(positional) not in (len(data_args), len(data_args) + 1):
        raise SignalConfigError(
            f"{name}: the function takes ({', '.join(positional)}); {usage}.",
            value=name,
        )
    return len(positional) == len(data_args) + 1, fields


def _fit_options(fit: Callable[..., Any] | None, options: Iterable[str]):
    if fit is None:
        return None
    if not callable(fit):
        raise SignalConfigError(f"fit must be a function, not {fit!r}.")
    params = inspect.signature(fit).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    names = {p.name for p in params if p.kind is p.KEYWORD_ONLY}
    return tuple(o for o in options if o in names)


def _register(
    role: Role,
    name: str,
    fn: Callable[..., Any],
    base: type,
    *,
    fit: Callable[..., Any] | None,
    flag: Flag | None = None,
) -> type:
    source = _source(fn)
    replace = _check_name(name, role, source)
    takes_state, fields = _signature(fn, role, name)
    if fit is not None and not takes_state:
        raise SignalConfigError(
            f"{name}: fit= is given, so the function must take the fitted state "
            f"after its data: fn({', '.join(_DATA_ARGS[role])}, state, *, ...).",
            value=name,
        )
    method = Method(
        role=role,
        name=name,
        function=fn,
        fit=fit,
        takes_state=takes_state,
        options=tuple(fields),
        fit_options=_fit_options(fit, fields),
        source=source,
        flag=flag,
    )
    cls = create_model(
        name,
        __base__=base,
        __module__=source["module"],
        __doc__=_summary(fn, role, name),
        kind=(Literal[name], Field(name, description="Which step this is.")),
        **fields,
    )
    cls.registered = method
    return register_step(cls, replace=replace)


# -- the registered steps -------------------------------------------------------------


def _rebuild(kind: str, data: dict[str, Any]) -> Any:
    return get_step_class(kind).model_validate(data)


class _Registered:
    """What every registered step shares: its method, options and pickling."""

    registered: ClassVar[Method]

    def _options(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.registered.options}

    def _fit_call(self, *args: Any) -> dict[str, Any]:
        method = self.registered
        options = self._options()
        if method.fit_options is not None:
            options = {k: options[k] for k in method.fit_options}
        state = method.fit(*args, **options)
        if not isinstance(state, Mapping):
            raise SignalConfigError(
                f"{method.name}: fit returned {type(state).__name__}; it must return "
                "a dict that saves as JSON."
            )
        return dict(state)

    def _call(self, *args: Any) -> Any:
        return self.registered.function(*args, **self._options())

    def __reduce__(self):
        """Pickle by kind, so a worker with the same methods registered unpickles it."""
        return _rebuild, (self.kind, self.model_dump(mode="json"))


def _as_array(out: Any, index: pd.DatetimeIndex, what: str) -> np.ndarray:
    if isinstance(out, pd.DataFrame):
        if out.shape[1] != 1:
            raise SignalDataError(
                f"{what} returned a table with {out.shape[1]} columns; return one "
                "value per row (a Series or an array)."
            )
        out = out.iloc[:, 0]
    if isinstance(out, pd.Series):
        if len(out) == len(index) and not out.index.equals(index):
            if isinstance(out.index, pd.DatetimeIndex):
                raise SignalDataError(
                    f"{what} returned a Series on a different time index; keep the "
                    "index it was given."
                )
        out = out.to_numpy()
    array = np.asarray(out)
    if array.shape != (len(index),):
        raise SignalDataError(
            f"{what} returned shape {array.shape}; it must return one value per row "
            f"({len(index)})."
        )
    return array


class RegisteredCleaner(_Registered, StreamStep):
    """Base of the steps :func:`register_cleaner` makes."""

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Learn each stream's state with the method's ``fit``, if it has one."""
        if self.registered.fit is None:
            return {}
        check_streams(sf, self.streams)
        values = sf.values
        return {"streams": {c: self._fit_call(values[c].copy()) for c in self.streams}}

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Reject the values the method returns True for."""
        check_streams(sf, self.streams)
        values = sf.values
        remove = {}
        for col in self.streams:
            args: list[Any] = [values[col].copy()]
            if self.registered.takes_state:
                args.append(dict(state.get("streams", {}).get(col, {})))
            mask = _as_array(self._call(*args), sf.index, f"{self.kind} on {col!r}")
            if mask.dtype != bool:
                try:
                    as_float = mask.astype(float)
                except (TypeError, ValueError):
                    as_float = np.array([np.nan])
                if np.isnan(as_float).any() or not np.isin(as_float, (0, 1)).all():
                    raise SignalDataError(
                        f"{self.kind} on {col!r} must return True/False per value, "
                        f"got {mask.dtype} values."
                    )
                mask = as_float.astype(bool)
            remove[col] = mask
        return remove_values(
            sf, pd.DataFrame(remove, index=sf.index), self.registered.flag
        )


class RegisteredImputer(_Registered, _GapImputer):
    """Base of the steps :func:`register_imputer` makes."""

    def _frame(self, sf: SignalFrame, stream: str) -> pd.DataFrame:
        columns = [stream, *(c for c in sf.columns if c != stream)]
        return sf.values[columns].copy()

    def fit(self, sf: SignalFrame) -> dict[str, Any]:
        """Learn each stream's state with the method's ``fit``, if it has one."""
        if self.registered.fit is None:
            return {}
        check_streams(sf, self.streams)
        return {
            "streams": {c: self._fit_call(self._frame(sf, c)) for c in self.streams}
        }

    def transform(self, sf: SignalFrame, state: Mapping[str, Any]) -> SignalFrame:
        """Fill the gaps with the values the method proposes."""
        targets = self._targets(sf)
        filled = {}
        for col, where in targets.items():
            if not where.any():
                continue
            args: list[Any] = [
                self._frame(sf, col),
                pd.Series(where, index=sf.index, name=col),
            ]
            if self.registered.takes_state:
                args.append(dict(state.get("streams", {}).get(col, {})))
            proposed = _as_array(self._call(*args), sf.index, f"{self.kind} on {col!r}")
            try:
                proposed = proposed.astype("float64")
            except (TypeError, ValueError):
                raise SignalDataError(
                    f"{self.kind} on {col!r} must return numbers, got "
                    f"{proposed.dtype} values."
                ) from None
            filled[col] = (where, proposed)
        return _fill(sf, filled, Flag.IMPUTED_MODEL)


class RegisteredForecaster(_Registered, ForecastStep):
    """Base of the steps :func:`register_forecaster` makes."""

    def _history(self, window: _Window, stream: str) -> pd.DataFrame:
        columns = [stream, *self.exog]
        return window.frame.values[columns].iloc[: window.n_history].copy()

    def _future(self, window: _Window) -> pd.DataFrame:
        return window.frame.values[list(self.exog)].iloc[window.future].copy()

    def _fit_one(self, window: _Window, stream: str) -> dict[str, Any]:
        if self.registered.fit is None:
            return {}
        return self._fit_call(self._history(window, stream))

    def _predict(self, window, stream, state):
        future = self._future(window)
        args: list[Any] = [self._history(window, stream), future]
        if self.registered.takes_state:
            args.append(dict(state))
        out = self._call(*args)
        what = f"{self.kind} on {stream!r}"
        if not isinstance(out, Mapping | pd.DataFrame):
            if self.quantiles:
                raise SignalDataError(
                    f"{what} returned only the mean, but quantiles are requested; "
                    "return {'mean': ..., 'std': ...} or {'mean': ..., 'q10': ...}."
                )
            return _as_array(out, future.index, what).astype("float64"), None
        if "mean" not in out:
            raise SignalDataError(
                f"{what} returned {list(out)}; a table or dict needs a 'mean' entry."
            )
        mean = _as_array(out["mean"], future.index, what).astype("float64")
        if not self.quantiles:
            return mean, None
        found = {}
        for q in self.quantiles:
            key = quantile_column("", q).lstrip("_")
            for candidate in (key, q):
                if candidate in out:
                    found[q] = _as_array(out[candidate], future.index, what).astype(
                        "float64"
                    )
        if len(found) == len(self.quantiles):
            return mean, found
        if "std" in out:
            return mean, _as_array(out["std"], future.index, what).astype("float64")
        missing = [quantile_column("", q).lstrip("_") for q in self.quantiles]
        raise SignalDataError(
            f"{what} returned no {missing} and no 'std', so the requested quantiles "
            "can't be written."
        )


# -- the public functions -------------------------------------------------------------


def register_cleaner(
    name: str,
    fn: Callable[..., Any],
    *,
    flag: str | Flag = "OUTLIER",
    fit: Callable[..., Any] | None = None,
) -> type:
    """Register a function that says which values to reject, as a cleaning step.

    The function is called once per stream as ``fn(values, *, **options)``, or
    ``fn(values, state, *, **options)`` with ``fit``. ``values`` is the stream as
    a pandas Series on the data's time index, NaN where a value is missing. It
    returns True for each value to reject (a boolean Series or array of the same
    length). flex-signal sets those values to NaN and adds ``flag``; values that
    are already missing are left alone.

    Args:
        name: The step's kind in JSON and its name in ``fs.clean``, e.g.
            ``"PlantDespike"``. A Python identifier that isn't a built-in step.
        fn: The function. Its keyword-only parameters, with type hints and
            defaults, become the step's options.
        flag: Why the values are rejected: ``OUTLIER`` (default),
            ``OUT_OF_RANGE``, ``RATE_OF_CHANGE``, ``STUCK``, ``MANUAL`` or
            ``EXCLUDED``.
        fit: Optional ``fit(values, *, **options) -> dict``, called once per
            stream on the training data. Its result must save as JSON.

    Returns:
        The step class, also reachable as ``fs.clean.<name>``.

    Raises:
        SignalConfigError: If the name, flag or signature isn't usable.

    Examples:
        >>> import flexsignal as fs
        >>> def despike(values, *, k: float = 4.0):
        ...     dev = (values - values.median()).abs()
        ...     return dev > k * dev.median()
        >>> step_class = fs.register_cleaner("DocDespike", despike)
        >>> fs.clean.DocDespike("flow", k=5.0).k
        5.0
    """
    if isinstance(flag, str):
        try:
            flag = Flag[flag]
        except KeyError:
            flag = None  # type: ignore[assignment]
    if flag not in CLEANER_FLAGS:
        allowed = ", ".join(f.name for f in CLEANER_FLAGS)
        raise SignalConfigError(
            f"a cleaner's flag must be one of {allowed}.", field="flag", value=flag
        )
    return _register("cleaner", name, fn, RegisteredCleaner, fit=fit, flag=flag)


def register_imputer(
    name: str, fn: Callable[..., Any], *, fit: Callable[..., Any] | None = None
) -> type:
    """Register a function that proposes values for gaps, as an imputation step.

    The function is called once per stream that has gaps to fill, as
    ``fn(values, gaps, *, **options)``, or ``fn(values, gaps, state, *, ...)``
    with ``fit``:

    - ``values`` is a DataFrame of every stream on the data's time index, with
      the stream being filled as the first column (the others are there to use
      as inputs);
    - ``gaps`` is a boolean Series, named after that stream, True where a value
      may be filled.

    It returns proposed values for the whole column (a Series or array); only
    the positions in ``gaps`` are used, and NaN proposals are skipped.
    flex-signal finds the gaps (no longer than the step's ``max_gap``, and never
    values flagged ``EXCLUDED`` or ``NO_SAMPLE``) and flags filled values
    ``IMPUTED_MODEL``. A registered imputer can be a ``tiered`` tier by name.

    Args:
        name: The step's kind in JSON and its name in ``fs.impute``.
        fn: The function. Its keyword-only parameters become the step's options;
            ``max_gap`` is added by flex-signal.
        fit: Optional ``fit(values, *, **options) -> dict``, called once per
            stream on the training data (``values`` as above). Its result must
            save as JSON.

    Returns:
        The step class, also reachable as ``fs.impute.<name>``.

    Raises:
        SignalConfigError: If the name or signature isn't usable.
    """
    return _register("imputer", name, fn, RegisteredImputer, fit=fit)


def register_forecaster(
    name: str, fn: Callable[..., Any], *, fit: Callable[..., Any] | None = None
) -> type:
    """Register a function that forecasts the horizon, as a forecast step.

    The function is called once per target stream as
    ``fn(history, future, *, **options)``, or ``fn(history, future, state, ...)``
    with ``fit``:

    - ``history`` is a DataFrame up to and including the issue time: the target
      as its first column, then the step's ``exog`` streams;
    - ``future`` is a DataFrame of the ``exog`` streams over the forecast rows
      (it has no columns without ``exog``); its index is the forecast's times.

    It returns one value per row of ``future``: an array or Series of means, or
    a dict (or DataFrame) with ``"mean"`` and, when the step asks for
    ``quantiles``, either ``"std"`` or one entry per quantile (``"q10"``, ...).
    flex-signal sets the issue time, writes the rows, flags them ``FORECAST``,
    writes the quantile streams, and makes the step work in
    :func:`~flexsignal.eval.backtest`.

    Args:
        name: The step's kind in JSON and its name in ``fs.forecast``.
        fn: The function. Its keyword-only parameters become the step's options;
            ``horizon``, ``issue_time``, ``exog`` and ``quantiles`` are added by
            flex-signal.
        fit: Optional ``fit(history, *, **options) -> dict``, called once per
            target on the training data. Its result must save as JSON; a
            ``"metrics"`` entry goes into the fit report.

    Returns:
        The step class, also reachable as ``fs.forecast.<name>``.

    Raises:
        SignalConfigError: If the name or signature isn't usable.
    """
    return _register("forecaster", name, fn, RegisteredForecaster, fit=fit)


def registered_methods() -> dict[str, Method]:
    """Return every registered method keyed by name (the step's kind), sorted."""
    return {
        kind: cls.registered
        for kind, cls in registered_steps().items()
        if isinstance(getattr(cls, "registered", None), Method)
    }


def registered_class(role: str, name: str) -> type | None:
    """Return the step class registered as ``name`` for ``role``, or None."""
    if name.startswith("_"):
        return None
    load_entry_points()
    cls = find_step(name)
    method = getattr(cls, "registered", None)
    if isinstance(method, Method) and method.role == role:
        return cls
    return None


def load_plugins(*modules: str) -> list[str]:
    """Import modules that register methods, chosen by the caller.

    The module names come from your code, never from a spec: a spec's
    ``requires`` list is only used in error messages.

    Args:
        modules: Module names, e.g. ``"myplant.methods"``.

    Returns:
        The step kinds and model names the modules registered, sorted.

    Raises:
        SignalConfigError: If a module can't be imported.
    """
    names: list[str] = []
    for item in modules:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, Iterable):
            names.extend(item)
        else:
            raise SignalConfigError(f"plugins are module names, not {item!r}.")
    before = set(registered_steps()) | set(registered_models())
    for module in names:
        if not isinstance(module, str) or not all(
            part.isidentifier() for part in module.split(".")
        ):
            raise SignalConfigError(
                f"plugin {module!r} isn't a module name, e.g. myplant.methods.",
                value=module,
            )
        try:
            import_module(module)
        except ImportError as exc:
            raise SignalConfigError(
                f"plugin module {module!r} could not be imported: {exc}. Install the "
                "package that provides it.",
                value=module,
            ) from exc
    after = set(registered_steps()) | set(registered_models())
    return sorted(after - before)
