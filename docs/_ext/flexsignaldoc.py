"""Sphinx directives that generate reference tables from flex-signal's own registries.

This is the counterpart of flex-pse's ``docs/_ext/flexdoc.py``. Generating
tables from the code means a reference page can never drift from what the
code actually registers.

``.. flexsignal-step-table::``
    Every registered step ``kind`` with its group, class, and summary, and
    whether it learns state in ``fit``.

``.. flexsignal-spec-table:: <dotted pydantic model path>``
    A step or spec model's fields: name, type, default, description.

``.. flexsignal-flag-table::``
    The quality-flag bits, what each means, and which steps set it.

``.. flexsignal-backend-table::``
    Every registered backend key, what it resolves to, and its extra.

Each directive wraps a plain ``collect_*`` function with no Sphinx dependency,
so ``tests/test_docs.py`` can test the collection logic directly.
"""

from __future__ import annotations

import importlib
import types
import typing

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

import flexsignal  # noqa: F401  (registers the built-in steps)
from flexsignal.backends import registered_backends
from flexsignal.core import FLAG_DESCRIPTIONS
from flexsignal.core.registry import registered_steps
from flexsignal.spec import StepSpec


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n\n")[0].replace("\n", " ")


def _type_name(annotation) -> str:
    if annotation is None or annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _type_name(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        return " | ".join(repr(a) for a in typing.get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        return " | ".join(_type_name(a) for a in typing.get_args(annotation))
    if origin in (tuple, list):
        args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        inner = _type_name(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin is dict:
        key, value = typing.get_args(annotation)
        return f"dict[{_type_name(key)}, {_type_name(value)}]"
    return getattr(annotation, "__name__", str(annotation))


def collect_step_rows() -> list[list[str]]:
    """Return ``[kind, group, class, learns state, summary]`` per registered step."""
    rows = []
    for kind, cls in registered_steps().items():
        group = cls.__module__.rsplit(".", 1)[-1]
        stateful = "yes" if cls.fit is not StepSpec.fit else "no"
        rows.append(
            [
                f"``{kind}``",
                group,
                f":class:`~{cls.__module__}.{cls.__qualname__}`",
                stateful,
                _first_line(cls.__doc__),
            ]
        )
    return rows


def collect_spec_rows(model: type[BaseModel]) -> list[list[str]]:
    """Return ``[field, type, default, description]`` per field of a pydantic model."""
    rows = []
    for name, field in model.model_fields.items():
        if field.is_required():
            default = "required"
        elif field.default_factory is not None:
            default = f"``{field.default_factory()!r}``"
        elif field.default is PydanticUndefined:
            default = "required"
        else:
            default = f"``{field.default!r}``"
        rows.append(
            [
                f"``{name}``",
                f"``{_type_name(field.annotation)}``",
                default,
                field.description or "",
            ]
        )
    return rows


def collect_flag_rows() -> list[list[str]]:
    """Return ``[flag, bit, meaning, set by]`` per flag."""
    return [
        [f"``{flag.name}``", str(flag.value), meaning, set_by]
        for flag, (meaning, set_by) in FLAG_DESCRIPTIONS.items()
    ]


def collect_backend_rows() -> list[list[str]]:
    """Return ``[key, target, extra, description]`` per registered backend."""
    rows = []
    for info in registered_backends():
        target = (
            f"`{info.target} <{info.docs_url}>`__"
            if info.docs_url
            else f"``{info.target}``"
        )
        rows.append([f"``{info.key}``", target, info.extra or "core", info.description])
    return rows


def _list_table(header: list[str], rows: list[list[str]], widths: str) -> StringList:
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    for row in [header, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {cell}" for cell in row[1:])
    return StringList(lines)


class _TableDirective(Directive):
    has_content = False

    def _render(self, content: StringList) -> list[nodes.Node]:
        container = nodes.container()
        self.state.nested_parse(content, self.content_offset, container)
        return container.children


class StepTable(_TableDirective):
    """``.. flexsignal-step-table::``"""

    def run(self):
        """Render the step table."""
        header = ["kind", "group", "class", "learns state", "summary"]
        return self._render(_list_table(header, collect_step_rows(), "12 8 20 8 52"))


class SpecTable(_TableDirective):
    """``.. flexsignal-spec-table:: dotted.path.Model``"""

    required_arguments = 1

    def run(self):
        """Render one model's field table."""
        module_name, _, attr = self.arguments[0].rpartition(".")
        model = getattr(importlib.import_module(module_name), attr)
        header = ["field", "type", "default", "description"]
        return self._render(
            _list_table(header, collect_spec_rows(model), "14 22 14 50")
        )


class FlagTable(_TableDirective):
    """``.. flexsignal-flag-table::``"""

    def run(self):
        """Render the flag table."""
        header = ["flag", "bit value", "meaning", "set by"]
        return self._render(_list_table(header, collect_flag_rows(), "16 8 50 26"))


class BackendTable(_TableDirective):
    """``.. flexsignal-backend-table::``"""

    def run(self):
        """Render the backend table."""
        header = ["key", "resolves to", "extra", "description"]
        return self._render(_list_table(header, collect_backend_rows(), "22 34 8 36"))


def setup(app):
    """Register the flex-signal directives."""
    app.add_directive("flexsignal-step-table", StepTable)
    app.add_directive("flexsignal-spec-table", SpecTable)
    app.add_directive("flexsignal-flag-table", FlagTable)
    app.add_directive("flexsignal-backend-table", BackendTable)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
