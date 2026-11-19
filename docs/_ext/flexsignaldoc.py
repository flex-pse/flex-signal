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

``.. flexsignal-model-table::``
    The adapters whose model option takes any of a library's models: the
    tested ones, and where to find the rest.

``.. flexsignal-method-table:: clean|impute|forecast``
    The included methods of one group: name, ``fs.`` call (linked to its API
    reference), JSON kind, extra, and links to the library docs.

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
from flexsignal.adapters._common import MODEL_OPTIONS
from flexsignal.backends import registered_backends
from flexsignal.backends.models import registered_models
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
    groups = {"cleaner": "clean", "imputer": "impute", "forecaster": "forecast"}
    for kind, cls in registered_steps().items():
        method = getattr(cls, "registered", None)
        if method is not None:
            # A registered method (an adapter, or a user's): marked as such, and
            # pointing at the function behind it.
            source = f"{method.source['module']}.{method.source['function']}"
            rows.append(
                [
                    f"``{kind}``",
                    groups[method.role],
                    f"registered: ``{source}``",
                    "yes" if method.fit is not None else "no",
                    _first_line(cls.__doc__),
                ]
            )
            continue
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
    for name, description in registered_models().items():
        rows.append([f"``{name}``", "registered model", "—", description])
    return rows


# The included methods, by group: a readable name, and the backend keys of the
# library functions each calls ("pyod:*" means every registered pyod key; an
# empty tuple means flex-signal implements it itself). The upstream links come
# from the backend registry, so they are written once. test_docs checks that
# every cleaner, imputer and forecaster flex-signal ships is listed here.
_LINEAR = "linear models"
METHODS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "clean": {
        "range": ("Range check", ()),
        "rate_of_change": ("Rate-of-change check", ()),
        "stuck": ("Stuck-sensor check", ()),
        "hampel": ("Hampel filter", ()),
        "tukey": ("Tukey's fences", ()),
        "expression": ("Rule written as an expression", ()),
        "filter": (
            "Smoothing: Savitzky-Golay, Butterworth or median",
            ("scipy:savgol_filter", "scipy:butter", "scipy:sosfiltfilt"),
        ),
        "detotalize": ("Running total to rate", ()),
        "PyodDetector": ("Outlier detectors (pyod)", ("pyod:*",)),
        "RupturesChangepoint": (
            "Level glitches between change points",
            ("ruptures:*",),
        ),
        "StumpyDiscord": ("Matrix-profile discords", ("stumpy:*",)),
    },
    "impute": {
        "interpolate": (
            "Interpolation: linear, nearest, previous, PCHIP or Akima",
            (
                "numpy:interp",
                "scipy:interp1d",
                "scipy:PchipInterpolator",
                "scipy:Akima1DInterpolator",
            ),
        ),
        "kalman": (
            "Kalman smoother of a structural model",
            ("statsmodels:UnobservedComponents",),
        ),
        "regression": ("Regression on other streams and time features", (_LINEAR,)),
        "knn": ("k-nearest neighbours", ("sklearn:KNNImputer",)),
        "iterative": ("Iterative, MICE-style", ("sklearn:IterativeImputer",)),
        "tiered": ("Tiered by gap length (routes to the others)", ()),
        "PypotsImpute": (
            "Deep imputation: SAITS, BRITS, CSDI, TimesNet",
            ("pypots:imputation.*",),
        ),
    },
    "forecast": {
        "forecast_persistence": ("Persistence", ()),
        "forecast_seasonal_naive": ("Seasonal naive", ()),
        "forecast_structural": (
            "Structural time-series model",
            ("statsmodels:UnobservedComponents",),
        ),
        "forecast_regression": ("Regression on cycles, exog and lags", (_LINEAR,)),
        "ChronosForecast": (
            "Chronos-2, zero-shot (or TimesFM 2.5)",
            ("chronos:*", "timesfm:*"),
        ),
        "T0Forecast": ("t0-alpha, zero-shot", ("t0:*",)),
        "PypotsForecast": ("Deep forecasting (PyPOTS)", ("pypots:forecasting.*",)),
        "NeuralforecastForecast": (
            "Deep forecasting (neuralforecast)",
            ("neuralforecast:*",),
        ),
        "DartsForecast": ("darts models", ("darts:*",)),
        "SktimeForecast": ("sktime forecasters", ("sktime:*",)),
    },
}


def method_group(cls) -> str | None:
    """Return ``clean``, ``impute`` or ``forecast`` for a method's step class."""
    method = getattr(cls, "registered", None)
    if method is not None:
        return {"cleaner": "clean", "imputer": "impute", "forecaster": "forecast"}[
            method.role
        ]
    group = cls.__module__.rsplit(".", 1)[-1]
    return group if group in METHODS else None


def _backend_keys(pattern: str) -> list[str]:
    from flexsignal.backends.models import LINEAR_MODELS

    keys = [info.key for info in registered_backends()]
    if pattern == _LINEAR:
        return [k for k in LINEAR_MODELS if k in keys]
    if pattern.endswith("*"):
        return [k for k in keys if k.startswith(pattern[:-1])]
    return [pattern] if pattern in keys else []


def collect_method_rows(group: str) -> list[list[str]]:
    """Return ``[method, call, kind, extra, library]`` per method in a group."""
    backends = {info.key: info for info in registered_backends()}
    steps = registered_steps()
    rows = []
    for kind, (name, patterns) in METHODS[group].items():
        cls = steps[kind]
        if getattr(cls, "registered", None) is not None:
            call = f":class:`fs.{group}.{kind} <flexsignal.steps.{group}.{kind}>`"
        else:
            call = (
                f":class:`fs.{group}.{cls.__qualname__} "
                f"<{cls.__module__}.{cls.__qualname__}>`"
            )
        keys = [k for p in patterns for k in _backend_keys(p)]
        extras = sorted({backends[k].extra for k in keys if backends[k].extra})
        by_library: dict[str, list[str]] = {}
        for key in keys:
            library, _, member = key.partition(":")
            info = backends[key]
            label = member.split(".")[-1] or library
            link = f"`{label} <{info.docs_url}>`__" if info.docs_url else label
            by_library.setdefault(library, []).append(link)
        library = (
            "; ".join(f"{lib}: {', '.join(links)}" for lib, links in by_library.items())
            or "flex-signal (no external library)"
        )
        rows.append(
            [
                name,
                call,
                f"``{kind}``",
                ", ".join(f"``[{e}]``" for e in extras) or "core",
                library,
            ]
        )
    return rows


def collect_model_rows() -> list[list[str]]:
    """Return ``[call, option, tested, other models, note]`` per adapter option."""
    steps = registered_steps()
    groups = {"cleaner": "clean", "imputer": "impute", "forecaster": "forecast"}
    rows = []
    for option in MODEL_OPTIONS:
        group = groups[steps[option.method].registered.role]
        call = (
            f":class:`fs.{group}.{option.method} "
            f"<flexsignal.steps.{group}.{option.method}>`"
        )
        rows.append(
            [
                call,
                f"``{option.option}``",
                ", ".join(f"``{name}``" for name in option.tested),
                f"`every {option.library} model <{option.catalogue}>`__, by class "
                f"name; settings in ``{option.params}``",
                option.note,
            ]
        )
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


class MethodTable(_TableDirective):
    """``.. flexsignal-method-table:: clean|impute|forecast``"""

    required_arguments = 1

    def run(self):
        """Render one group's included methods."""
        header = ["method", "flex-signal call", "JSON kind", "extra", "library docs"]
        rows = collect_method_rows(self.arguments[0])
        return self._render(_list_table(header, rows, "24 22 16 10 28"))


class ModelTable(_TableDirective):
    """``.. flexsignal-model-table::``"""

    def run(self):
        """Render the adapters' model options."""
        header = ["step", "option", "tested", "other models", "note"]
        rows = collect_model_rows()
        return self._render(_list_table(header, rows, "20 10 26 24 20"))


def setup(app):
    """Register the flex-signal directives."""
    app.add_directive("flexsignal-model-table", ModelTable)
    app.add_directive("flexsignal-method-table", MethodTable)
    app.add_directive("flexsignal-step-table", StepTable)
    app.add_directive("flexsignal-spec-table", SpecTable)
    app.add_directive("flexsignal-flag-table", FlagTable)
    app.add_directive("flexsignal-backend-table", BackendTable)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
