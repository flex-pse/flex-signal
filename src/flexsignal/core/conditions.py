"""Evaluate condition expressions over a SignalFrame's streams.

Conditions are small arithmetic and boolean expressions over stream names, for
example ``"C == 0 and D == 0 and B > 0"`` or
``"abs(pressure - (4 + 0.01 * flow)) > 0.15"``. Names that aren't Python
identifiers go in backticks: ```flow rate` > 5``. A comparison with a missing
(NaN) value is False.

A condition can only *read* streams, never run code. Before anything is
evaluated, :func:`~flexsignal.core.check_condition` parses it and accepts only:

- stream names, numbers, ``True`` and ``False``;
- arithmetic ``+ - * / // % **`` (``**`` only with a number as the exponent);
- comparisons ``== != < <= > >=``, and ``and or not & | ~``;
- the math functions in :data:`~flexsignal.core.CONDITION_FUNCTIONS`, e.g. ``abs(x)``.

Everything else is refused, including attribute access (``x.y``), indexing
(``x[0]``), strings, keyword arguments, other function names and ``@``
variables. So a spec from someone else can't use a condition to reach Python
internals. This check is what makes conditions safe; ``pandas.eval`` on its
own is not a security boundary.

M2 adds duration and flag helpers (``held(...)``, ``is_raw(...)``,
``flagged(...)``) on top of this.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from typing import Annotated

import pandas as pd
from pydantic import AfterValidator

from flexsignal.core.exceptions import SignalConfigError

CONDITION_FUNCTIONS = frozenset(
    {
        "abs",
        "sqrt",
        "exp",
        "expm1",
        "log",
        "log10",
        "log1p",
        "floor",
        "ceil",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "arctan2",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
    }
)
"""Functions a condition may call (each also supported by ``pandas.eval``)."""

MAX_CONDITION_LENGTH = 1000
"""Longest condition accepted, in characters."""

MAX_EXPONENT = 100
"""Largest exponent allowed after ``**``, so a condition can't be made to hang."""

_BACKTICKED = re.compile(r"`[^`]*`")
_BOOL_OPS = (ast.And, ast.Or)
_UNARY_OPS = (ast.Not, ast.USub, ast.UAdd, ast.Invert)
_BIN_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
)
_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


def check_condition(expression: str, streams: Iterable[str] | None = None) -> str:
    """Check that a condition uses only the allowed syntax.

    Args:
        expression: The condition, e.g. ``"pump_on == 1 and flow > 0"``.
        streams: Stream names it may refer to. If None, any name that isn't a
            function is accepted (used when a spec is loaded, before the data is
            known).

    Returns:
        ``expression`` unchanged.

    Raises:
        ValueError: Saying what is not allowed. (:func:`evaluate_condition` and
            spec loading turn it into a ``SignalConfigError`` with the field.)
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("a condition must be a non-empty string, e.g. 'flow > 0'.")
    if len(expression) > MAX_CONDITION_LENGTH:
        raise ValueError(
            f"the condition is {len(expression)} characters long; the limit is "
            f"{MAX_CONDITION_LENGTH}."
        )
    if "@" in expression:
        raise ValueError(
            "conditions may refer only to stream names; '@' variable references "
            "are not allowed."
        )
    # Backticked names may hold any characters; swap each for a plain placeholder.
    quoted: dict[str, str] = {}

    def _placeholder(match: re.Match[str]) -> str:
        name = f"_q{len(quoted)}_"
        quoted[name] = match.group(0)[1:-1]
        return name

    try:
        source = _BACKTICKED.sub(_placeholder, expression).strip()
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{expression!r} is not a valid condition: {exc.msg}."
        ) from exc
    known = None if streams is None else set(streams)
    for node in ast.walk(tree):
        _check_node(node, quoted, known)
    return expression


def _check_node(node: ast.AST, quoted: dict[str, str], known: set[str] | None) -> None:
    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in CONDITION_FUNCTIONS:
            shown = ast.unparse(func)
            raise ValueError(
                f"{shown!r} can't be called in a condition. Allowed functions: "
                f"{', '.join(sorted(CONDITION_FUNCTIONS))}."
            )
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError(
                f"{func.id}() takes plain arguments only, e.g. {func.id}(flow)."
            )
        return
    if isinstance(node, ast.Name):
        if node.id in CONDITION_FUNCTIONS or node.id in quoted:
            return
        if known is not None and node.id not in known:
            raise ValueError(
                f"unknown stream {node.id!r}; conditions may use these streams: "
                f"{sorted(known)} (put names that aren't Python identifiers in "
                "backticks)."
            )
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool | int | float):
            return
        raise ValueError(
            f"{node.value!r} is not allowed in a condition; use numbers, True or "
            "False."
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = node.right
        if isinstance(exponent, ast.UnaryOp) and isinstance(
            exponent.op, ast.USub | ast.UAdd
        ):
            exponent = exponent.operand
        if not (
            isinstance(exponent, ast.Constant)
            and isinstance(exponent.value, int | float)
            and not isinstance(exponent.value, bool)
            and abs(exponent.value) <= MAX_EXPONENT
        ):
            raise ValueError(
                f"the exponent after '**' must be a number no larger than "
                f"{MAX_EXPONENT}, e.g. 'flow ** 2'."
            )
        return
    allowed = (
        ast.Expression,
        ast.Load,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.Compare,
        *_BOOL_OPS,
        *_UNARY_OPS,
        *_BIN_OPS,
        *_COMPARE_OPS,
    )
    if not isinstance(node, allowed):
        what = {
            ast.Attribute: "attribute access ('.')",
            ast.Subscript: "indexing ('[...]')",
        }.get(type(node), f"'{type(node).__name__}'")
        raise ValueError(
            f"{what} is not allowed in a condition. Conditions may use stream "
            "names, numbers, arithmetic, comparisons, and/or/not and the math "
            "functions."
        )


Condition = Annotated[str, AfterValidator(check_condition)]
"""A spec field holding a condition, checked by :func:`check_condition` on load."""


def evaluate_condition(
    values: pd.DataFrame, expression: str, *, field: str | None = None
) -> pd.Series:
    """Evaluate a boolean expression row by row.

    The expression is checked with :func:`check_condition` first, so it can only
    read the streams in ``values``.

    Args:
        values: The streams the expression may refer to, by column name.
        expression: A condition that yields a boolean per row.
        field: Spec field path, used in error messages.

    Returns:
        A boolean Series aligned to ``values.index``. NaN results count as False.

    Raises:
        SignalConfigError: If the expression uses syntax that isn't allowed,
            refers to an unknown stream, or doesn't yield one boolean per row.
    """
    streams = [str(c) for c in values.columns]
    try:
        check_condition(expression, streams)
    except ValueError as exc:
        raise SignalConfigError(str(exc), field=field, value=expression) from exc
    try:
        result = values.eval(expression, engine="python")
    except Exception as exc:  # pandas raises many types for bad expressions
        raise SignalConfigError(
            f"could not evaluate condition {expression!r}: {exc}. Conditions may use "
            f"these streams: {list(values.columns)} (put names that aren't Python "
            "identifiers in backticks).",
            field=field,
            value=expression,
        ) from exc
    if not isinstance(result, pd.Series) or len(result) != len(values):
        raise SignalConfigError(
            f"condition {expression!r} must give one True/False value per row, e.g. "
            "'pump_on == 1'.",
            field=field,
            value=expression,
        )
    if not (pd.api.types.is_bool_dtype(result) or result.isna().all()):
        if not pd.api.types.is_object_dtype(result):
            raise SignalConfigError(
                f"condition {expression!r} gives {result.dtype} values, not "
                "True/False. "
                "Compare a stream to something, e.g. 'flow > 0'.",
                field=field,
                value=expression,
            )
    return result.fillna(False).astype(bool).set_axis(values.index)
