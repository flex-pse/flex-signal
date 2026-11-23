"""Conditions and formulas: small expressions over a SignalFrame's streams.

A **condition** gives True or False per row, for example
``"C == 0 and D == 0 and B > 0"`` or ``"abs(pressure - (4 + 0.01 * flow)) > 0.15"``.
A **formula** gives a number per row, for example ``"a + b * A"``; ``calibrate``
fits its parameters. Names that aren't Python identifiers go in backticks:
```flow rate` > 5``.

Both can only *read* streams, never run code. flex-signal parses every
expression itself and evaluates it with numpy, so there is no ``eval``. Only
this syntax is accepted:

- stream names, numbers, ``True`` and ``False``;
- arithmetic ``+ - * / // % **`` (``**`` only with a number as the exponent);
- comparisons ``== != < <= > >=``, which may be chained (``0 < flow <= 50``);
- ``and``, ``or``, ``not``, and their short forms ``&``, ``|``, ``~``;
- the math functions in :data:`~flexsignal.core.CONDITION_FUNCTIONS`, such as
  ``abs(x)``;
- in conditions, the helpers in :data:`~flexsignal.core.CONDITION_HELPERS`:

  - ``held(C == 0, "30min")``: the condition has been true without a break for
    at least 30 minutes;
  - ``valid(B)``: B has a value;
  - ``is_raw(B)``: B has a value that no step has touched (no flags at all);
  - ``flagged(A, "OUTLIER", "STUCK")``: A carries any of these flags.

Everything else is refused, including attribute access (``x.y``), indexing
(``x[0]``), keyword arguments, other function names, and strings outside the
helpers' arguments. So a spec from someone else can't use an expression to
reach Python internals.

Two rules differ from plain Python. As in ``pandas.eval``, ``&`` and ``|`` mean
``and`` and ``or`` with the same low precedence, so ``a > 1 & b < 2`` means
``(a > 1) and (b < 2)``. And every comparison with a missing value is False,
``!=`` included, so ``flow != 0`` is False where flow is missing.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import AfterValidator

from flexsignal.core.exceptions import SignalConfigError
from flexsignal.core.flags import Flag, parse_flags
from flexsignal.core.times import check_duration

__all__ = [
    "CONDITION_FUNCTIONS",
    "CONDITION_HELPERS",
    "Condition",
    "Formula",
    "check_condition",
    "check_formula",
    "evaluate_condition",
    "evaluate_formula",
]

_FUNCTIONS: dict[str, tuple[Callable[..., Any], int]] = {
    "abs": (np.abs, 1),
    "sqrt": (np.sqrt, 1),
    "exp": (np.exp, 1),
    "expm1": (np.expm1, 1),
    "log": (np.log, 1),
    "log10": (np.log10, 1),
    "log1p": (np.log1p, 1),
    "floor": (np.floor, 1),
    "ceil": (np.ceil, 1),
    "sin": (np.sin, 1),
    "cos": (np.cos, 1),
    "tan": (np.tan, 1),
    "arcsin": (np.arcsin, 1),
    "arccos": (np.arccos, 1),
    "arctan": (np.arctan, 1),
    "arctan2": (np.arctan2, 2),
    "sinh": (np.sinh, 1),
    "cosh": (np.cosh, 1),
    "tanh": (np.tanh, 1),
    "arcsinh": (np.arcsinh, 1),
    "arccosh": (np.arccosh, 1),
    "arctanh": (np.arctanh, 1),
    "minimum": (np.fmin, 2),
    "maximum": (np.fmax, 2),
}

CONDITION_FUNCTIONS = frozenset(_FUNCTIONS)
"""Math functions an expression may call, each applied element-wise with numpy.

``minimum`` and ``maximum`` take two arguments and ignore a missing one."""

CONDITION_HELPERS = frozenset({"held", "valid", "is_raw", "flagged"})
"""Helpers a condition may call: duration and flag predicates (see the module)."""

MAX_CONDITION_LENGTH = 1000
"""Longest expression accepted, in characters."""

MAX_EXPONENT = 100
"""Largest exponent allowed after ``**``, so an expression can't be made to hang."""

_BACKTICKED = re.compile(r"`[^`]*`")
_BOOL_OPS = {ast.And: np.logical_and, ast.Or: np.logical_or}
_BIN_OPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.FloorDiv: np.floor_divide,
    ast.Mod: np.mod,
    ast.Pow: np.power,
}
_COMPARE_OPS = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}

_Kind = Literal["bool", "number"]
_ALLOWED = (
    "Conditions and formulas may use stream names, numbers, arithmetic, "
    "comparisons, and/or/not, the math functions and, in conditions, held(), "
    "valid(), is_raw() and flagged()."
)


# -- parsing and checking ----------------------------------------------------------


class _Parsed:
    """A checked expression, ready to evaluate."""

    __slots__ = ("tree", "quoted", "names")

    def __init__(self, tree: ast.Expression, quoted: dict[str, str], names: set[str]):
        self.tree = tree
        self.quoted = quoted
        self.names = names


def _parse(expression: str) -> tuple[ast.Expression, dict[str, str]]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("an expression must be a non-empty string, e.g. 'flow > 0'.")
    if len(expression) > MAX_CONDITION_LENGTH:
        raise ValueError(
            f"the expression is {len(expression)} characters long; the limit is "
            f"{MAX_CONDITION_LENGTH}."
        )
    # Backticked names may hold any characters; swap each for a plain placeholder.
    quoted: dict[str, str] = {}

    def _placeholder(match: re.Match[str]) -> str:
        name = f"_q{len(quoted)}_"
        quoted[name] = match.group(0)[1:-1]
        return name

    source = _BACKTICKED.sub(_placeholder, expression).strip()
    if "@" in source:
        raise ValueError(
            "expressions may refer only to stream names; '@' variable references "
            "are not allowed (put a stream name containing '@' in backticks)."
        )
    # As in pandas.eval, & and | are and/or, with their low precedence.
    source = source.replace("&", " and ").replace("|", " or ")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{expression!r} is not a valid expression: {exc.msg}."
        ) from exc
    return tree, quoted


class _Checker:
    """Walks a parsed expression, refusing anything not allowed, and infers types."""

    def __init__(
        self,
        quoted: dict[str, str],
        streams: set[str] | None,
        parameters: set[str],
        helpers: bool,
    ):
        self.quoted = quoted
        self.streams = streams
        self.parameters = parameters
        self.helpers = helpers
        self.names: set[str] = set()

    def stream(self, node: ast.expr, helper: str) -> None:
        if not isinstance(node, ast.Name):
            raise ValueError(
                f"{helper}() takes a stream name as its first argument, e.g. "
                f"{helper}(flow)."
            )
        self.name(node.id)

    def name(self, ident: str) -> None:
        name = self.quoted.get(ident, ident)
        if ident in self.parameters:
            return
        if self.streams is not None and name not in self.streams:
            raise ValueError(
                f"unknown stream {name!r}; expressions may use these streams: "
                f"{sorted(self.streams)} (put names that aren't Python identifiers in "
                "backticks)."
            )
        self.names.add(name)

    def visit(self, node: ast.AST) -> _Kind:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Name):
            if node.id in CONDITION_FUNCTIONS or node.id in CONDITION_HELPERS:
                raise ValueError(
                    f"{node.id} is a function; call it, e.g. {node.id}(x)."
                )
            self.name(node.id)
            return "number"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int | float):
                return "number"
            raise ValueError(
                f"{node.value!r} is not allowed here; use numbers, True or False "
                "(strings appear only as held() and flagged() arguments)."
            )
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self.expect(value, "bool", "and/or")
            return "bool"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not | ast.Invert):
                self.expect(node.operand, "bool", "not/~")
                return "bool"
            if isinstance(node.op, ast.USub | ast.UAdd):
                self.expect(node.operand, "number", "a sign")
                return "number"
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            self.expect(node.left, "number", "arithmetic")
            if isinstance(node.op, ast.Pow):
                self.exponent(node.right)
            else:
                self.expect(node.right, "number", "arithmetic")
            return "number"
        if isinstance(node, ast.Compare):
            if not all(type(op) in _COMPARE_OPS for op in node.ops):
                raise ValueError(
                    "only the comparisons == != < <= > >= are allowed ('in' and 'is' "
                    "are not)."
                )
            kinds = {self.visit(x) for x in [node.left, *node.comparators]}
            if len(kinds) > 1:
                raise ValueError(
                    "a comparison mixes True/False and numbers; compare like with like."
                )
            return "bool"
        if isinstance(node, ast.Call):
            return self.call(node)
        what = {
            ast.Attribute: "attribute access ('.')",
            ast.Subscript: "indexing ('[...]')",
            ast.IfExp: "'x if c else y'",
            ast.Lambda: "lambda",
            ast.NamedExpr: "':='",
        }.get(type(node), f"'{type(node).__name__}'")
        raise ValueError(f"{what} is not allowed in an expression. {_ALLOWED}")

    def expect(self, node: ast.AST, kind: _Kind, where: str) -> None:
        got = self.visit(node)
        if got != kind:
            shown = ast.unparse(node)
            if kind == "bool":
                raise ValueError(
                    f"{where} needs True/False, but {shown!r} is a number; compare "
                    f"it, e.g. '{shown} > 0'."
                )
            raise ValueError(f"{where} needs numbers, but {shown!r} is True/False.")

    def exponent(self, node: ast.AST) -> None:
        value = node
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub | ast.UAdd):
            value = value.operand
        if not (
            isinstance(value, ast.Constant)
            and isinstance(value.value, int | float)
            and not isinstance(value.value, bool)
            and abs(value.value) <= MAX_EXPONENT
        ):
            raise ValueError(
                f"the exponent after '**' must be a number no larger than "
                f"{MAX_EXPONENT}, e.g. 'flow ** 2'."
            )

    def call(self, node: ast.Call) -> _Kind:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        allowed = CONDITION_FUNCTIONS | (CONDITION_HELPERS if self.helpers else set())
        if name not in allowed:
            shown = ast.unparse(func)
            extra = (
                " held(), valid(), is_raw() and flagged() work only in conditions."
                if name in CONDITION_HELPERS
                else ""
            )
            raise ValueError(
                f"{shown!r} can't be called here. Allowed functions: "
                f"{', '.join(sorted(allowed))}.{extra}"
            )
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError(f"{name}() takes plain arguments only, e.g. {name}(flow).")
        args = node.args
        if name in CONDITION_FUNCTIONS:
            n_args = _FUNCTIONS[name][1]
            if len(args) != n_args:
                raise ValueError(
                    f"{name}() takes {n_args} argument(s), got {len(args)}."
                )
            for arg in args:
                self.expect(arg, "number", f"{name}()")
            return "number"
        if name == "held":
            if len(args) != 2 or not _is_str(args[1]):
                raise ValueError(
                    "held() takes a condition and a duration, e.g. "
                    'held(C == 0, "30min").'
                )
            self.expect(args[0], "bool", "held()")
            _held_duration(args[1].value)  # type: ignore[attr-defined]
            return "bool"
        if name in ("valid", "is_raw"):
            if len(args) != 1:
                raise ValueError(f"{name}() takes one stream, e.g. {name}(flow).")
            self.stream(args[0], name)
            return "bool"
        # flagged(stream, "FLAG", ...)
        if len(args) < 2 or not all(_is_str(a) for a in args[1:]):
            raise ValueError(
                "flagged() takes a stream and one or more flag names, e.g. "
                'flagged(A, "OUTLIER", "STUCK").'
            )
        self.stream(args[0], "flagged")
        try:
            parse_flags([a.value for a in args[1:]])  # type: ignore[attr-defined]
        except ValueError as exc:
            raise ValueError(f"flagged(): {exc}.") from None
        return "bool"


def _is_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _held_duration(text: str) -> pd.Timedelta:
    try:
        return check_duration(text, allow_zero=True)
    except ValueError as exc:
        raise ValueError(f"held(): {exc}.") from None


def _check(
    expression: str,
    streams: Iterable[str] | None,
    *,
    kind: _Kind,
    parameters: Iterable[str] = (),
) -> _Parsed:
    tree, quoted = _parse(expression)
    checker = _Checker(
        quoted,
        None if streams is None else {str(s) for s in streams},
        set(parameters),
        helpers=kind == "bool",
    )
    got = checker.visit(tree)
    if got != kind:
        if kind == "bool":
            raise ValueError(
                f"condition {expression!r} gives numbers, not True/False. Compare "
                "it to something, e.g. 'flow > 0'."
            )
        raise ValueError(
            f"formula {expression!r} gives True/False, not numbers, e.g. 'a + b * x'."
        )
    return _Parsed(tree, quoted, checker.names)


def check_condition(expression: str, streams: Iterable[str] | None = None) -> str:
    """Check that a condition uses only the allowed syntax and gives True/False.

    Args:
        expression: The condition, e.g. ``"pump_on == 1 and flow > 0"``.
        streams: Stream names it may refer to. If None, any name is accepted
            (used when a spec is loaded, before the data is known).

    Returns:
        ``expression`` unchanged.

    Raises:
        ValueError: Saying what is not allowed. (:func:`evaluate_condition` and
            spec loading turn it into a ``SignalConfigError`` with the field.)
    """
    _check(expression, streams, kind="bool")
    return expression


def check_formula(
    expression: str,
    streams: Iterable[str] | None = None,
    parameters: Iterable[str] = (),
) -> str:
    """Check that a formula uses only the allowed syntax and gives numbers.

    Args:
        expression: The formula, e.g. ``"a + b * A"``.
        streams: Stream names it may refer to (None: any name).
        parameters: Names that are fitted parameters rather than streams.

    Returns:
        ``expression`` unchanged.

    Raises:
        ValueError: Saying what is not allowed.
    """
    _check(expression, streams, kind="number", parameters=parameters)
    return expression


def formula_names(expression: str, parameters: Iterable[str] = ()) -> list[str]:
    """Return the stream names a formula uses, sorted.

    Raises:
        ValueError: If the formula isn't allowed.
    """
    return sorted(_check(expression, None, kind="number", parameters=parameters).names)


Condition = Annotated[str, AfterValidator(check_condition)]
"""A spec field holding a condition, checked by :func:`check_condition` on load."""

Formula = Annotated[str, AfterValidator(check_formula)]
"""A spec field holding a formula, checked by :func:`check_formula` on load."""


# -- evaluation ----------------------------------------------------------------------


class _Evaluator:
    def __init__(
        self,
        values: pd.DataFrame,
        flags: pd.DataFrame | None,
        quoted: dict[str, str],
        parameters: Mapping[str, float],
    ):
        self.values = values
        self.flags = flags
        self.quoted = quoted
        self.parameters = parameters
        self.n = len(values)

    def column(self, ident: str) -> np.ndarray:
        return self.values[self.quoted.get(ident, ident)].to_numpy(dtype="float64")

    def flag_column(self, node: ast.expr, helper: str) -> np.ndarray:
        if self.flags is None:
            raise ValueError(
                f"{helper}() reads flags, so evaluate the condition on a SignalFrame, "
                "not a bare DataFrame."
            )
        name = node.id  # type: ignore[attr-defined]
        return self.flags[self.quoted.get(name, name)].to_numpy().astype("int64")

    def run(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.run(node.body)
        if isinstance(node, ast.Name):
            if node.id in self.parameters:
                return float(self.parameters[node.id])
            return self.column(node.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return node.value
            # Integer constants become floats, so they behave as in the
            # float64 data: no int64 overflow, no int ** -n error, x // 0 = inf.
            try:
                return float(node.value)
            except OverflowError:
                raise ValueError(
                    f"the number {ast.unparse(node)[:20]}... is too large; write "
                    "it in scientific notation, e.g. 1e30."
                ) from None
        if isinstance(node, ast.BoolOp):
            op = _BOOL_OPS[type(node.op)]
            result = self.run(node.values[0])
            for value in node.values[1:]:
                result = op(result, self.run(value))
            return result
        if isinstance(node, ast.UnaryOp):
            operand = self.run(node.operand)
            if isinstance(node.op, ast.Not | ast.Invert):
                return np.logical_not(operand)
            return np.negative(operand) if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            return _BIN_OPS[type(node.op)](self.run(node.left), self.run(node.right))
        if isinstance(node, ast.Compare):
            result: Any = True
            left = self.run(node.left)
            for op, right_node in zip(node.ops, node.comparators, strict=True):
                right = self.run(right_node)
                # Any comparison with a missing value is False (!= included).
                known = ~(_missing(left) | _missing(right))
                result = result & known & _COMPARE_OPS[type(op)](left, right)
                left = right
            return result
        if isinstance(node, ast.Call):
            return self.call(node)
        raise AssertionError(type(node))  # pragma: no cover - _Checker refuses it

    def call(self, node: ast.Call) -> Any:
        name = node.func.id  # type: ignore[attr-defined]
        if name in _FUNCTIONS:
            return _FUNCTIONS[name][0](*(self.run(a) for a in node.args))
        if name == "held":
            if not isinstance(self.values.index, pd.DatetimeIndex):
                raise ValueError(
                    "held() measures time, so the data needs timestamps: evaluate "
                    "the condition on a SignalFrame, or a DataFrame with a "
                    f"DatetimeIndex (got a {type(self.values.index).__name__})."
                )
            cond = np.broadcast_to(self.run(node.args[0]), (self.n,))
            delta = _held_duration(node.args[1].value)  # type: ignore[attr-defined]
            return _held(cond, self.values.index, delta)
        if name == "valid":
            return ~np.isnan(self.column(node.args[0].id))  # type: ignore[attr-defined]
        if name == "is_raw":
            flags = self.flag_column(node.args[0], name)
            column = self.column(node.args[0].id)  # type: ignore[attr-defined]
            return (flags == int(Flag.RAW)) & ~np.isnan(column)
        bits = int(parse_flags([a.value for a in node.args[1:]]))  # type: ignore[attr-defined]
        return (self.flag_column(node.args[0], name) & bits) != 0


def _missing(x: Any) -> Any:
    arr = np.asarray(x)
    return np.isnan(arr) if arr.dtype.kind == "f" else np.zeros(arr.shape, bool)


def _held(cond: np.ndarray, index: pd.DatetimeIndex, delta: pd.Timedelta) -> np.ndarray:
    """True where ``cond`` has been true, without a break, for at least ``delta``."""
    cond = np.asarray(cond, dtype=bool)
    if not len(cond):
        return cond
    positions = np.arange(len(cond))
    starts = cond & np.concatenate([[True], ~cond[:-1]])
    run_start = np.maximum.accumulate(np.where(starts, positions, 0))
    t = index.asi8
    return cond & ((t - t[run_start]) >= delta.value)


def _frame_parts(
    data: Any, field: str | None, expression: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    # A SignalFrame, duck-typed so this module doesn't need the frame module.
    if (
        hasattr(data, "values")
        and hasattr(data, "flags")
        and not isinstance(data, pd.DataFrame)
    ):
        return data.values, data.flags
    if isinstance(data, pd.DataFrame):
        return data, None
    raise SignalConfigError(
        f"expected a SignalFrame or DataFrame, got {type(data).__name__}.",
        field=field,
        value=expression,
    )


def evaluate_condition(
    data: Any, expression: str, *, field: str | None = None
) -> pd.Series:
    """Evaluate a condition row by row.

    The condition is checked with :func:`check_condition` first, so it can only
    read the streams in ``data``.

    Args:
        data: A :class:`~flexsignal.core.SignalFrame`, or a DataFrame of values.
            ``is_raw()`` and ``flagged()`` read flags, so they need a SignalFrame.
        expression: A condition that gives True or False per row.
        field: Spec field path, used in error messages.

    Returns:
        A boolean Series aligned to the data's index.

    Raises:
        SignalConfigError: If the condition uses syntax that isn't allowed, refers
            to an unknown stream, or doesn't give True/False.
    """
    values, flags = _frame_parts(data, field, expression)
    try:
        parsed = _check(expression, [str(c) for c in values.columns], kind="bool")
        with np.errstate(all="ignore"):
            result = _Evaluator(values, flags, parsed.quoted, {}).run(parsed.tree)
    except ValueError as exc:
        raise SignalConfigError(str(exc), field=field, value=expression) from exc
    result = np.broadcast_to(np.asarray(result, dtype=bool), (len(values),))
    return pd.Series(result.copy(), index=values.index)


def evaluate_formula(
    data: Any,
    expression: str,
    parameters: Mapping[str, float] | None = None,
    *,
    field: str | None = None,
) -> np.ndarray:
    """Evaluate a formula row by row.

    Args:
        data: A SignalFrame or a DataFrame of values.
        expression: A formula, e.g. ``"a + b * A"``.
        parameters: Values of the formula's parameters, e.g. ``{"a": 0, "b": 1}``.
        field: Spec field path, used in error messages.

    Returns:
        A float array, one value per row (NaN where an input is missing).

    Raises:
        SignalConfigError: If the formula isn't allowed or names an unknown stream.
    """
    values, _ = _frame_parts(data, field, expression)
    parameters = dict(parameters or {})
    try:
        parsed = _check(
            expression,
            [str(c) for c in values.columns],
            kind="number",
            parameters=parameters,
        )
        with np.errstate(all="ignore"):
            result = _Evaluator(values, None, parsed.quoted, parameters).run(
                parsed.tree
            )
    except ValueError as exc:
        raise SignalConfigError(str(exc), field=field, value=expression) from exc
    return np.broadcast_to(np.asarray(result, dtype="float64"), (len(values),)).copy()
