"""Sphinx configuration for flex-signal (mirrors flex-pse's docs/conf.py)."""

import sys
from importlib.metadata import version as _dist_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "flex-signal"
copyright = "2026, flex-signal contributors"
author = "flex-signal contributors"
release = _dist_version("flex-signal")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "flexsignaldoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_tutorials"]
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Docstrings name flex-signal's errors by their short names (``Raises:
# SignalDataError: ...``). Map them to where they are documented, so the
# cross-references resolve instead of being ignored.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    name: f"~flexsignal.core.{name}"
    for name in (
        "SignalError",
        "SignalConfigError",
        "SignalDataError",
        "SignalBackendError",
        "SignalWarning",
        "NaiveTimestampWarning",
    )
}

# Steps and spec models are pydantic models. Their generated constructor
# signature lists validator internals (annotated_types metadata, private
# check functions), which isn't useful to a reader and has no cross-reference
# targets. Keep class signatures off the class line; each step page shows a
# generated field table instead (flexsignal-spec-table).
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"

# Optional backends are mocked so the docs build needs only the [docs] extra.
# Add a module here when the backend that imports it lands (PLAN.md §7).
autodoc_mock_imports: list[str] = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "flexpse": ("https://flex-pse.github.io/flex-pse/", None),
}

# Every broken cross-reference fails the build (-W in CI). Add an entry to
# nitpick_ignore only with a comment explaining why it cannot resolve.
nitpicky = True
nitpick_ignore: list[tuple[str, str]] = []

myst_heading_anchors = 3

html_theme = "furo"
html_title = "flex-signal"
html_static_path = ["_static"]
# The marimo tutorials are exported to static HTML before the Sphinx build (see
# .github/workflows/docs.yml) into docs/_tutorials/tutorials/, and copied into
# the site next to tutorials/index.html. A local build without the export
# still works; only the links to the tutorial pages go nowhere.
if (Path(__file__).parent / "_tutorials").is_dir():
    html_extra_path = ["_tutorials"]


def _skip_deprecated_or_todo(app, what, name, obj, skip, options):
    """Drop members marked deprecated or TODO from the generated reference.

    A member is skipped when its docstring carries a Sphinx ``.. deprecated::``
    directive or begins with a ``TODO``/``DEPRECATED`` marker, so provisional
    or retired API never renders into the public docs. Returning ``None`` for
    everything else defers to autodoc's normal decision.
    """
    doc = (getattr(obj, "__doc__", None) or "").lstrip()
    if ".. deprecated::" in doc or doc.startswith(("TODO", "DEPRECATED")):
        return True
    return None


def setup(app):
    """Register the deprecated/TODO member filter."""
    app.connect("autodoc-skip-member", _skip_deprecated_or_todo)
