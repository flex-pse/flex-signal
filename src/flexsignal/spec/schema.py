"""Generate the SignalSpec JSON Schema from the registered step classes.

The schema is committed at ``src/flexsignal/spec/schemas/signal_spec.schema.json``,
and a test checks that it matches the code. Regenerate it with::

    python -m flexsignal.spec.schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Union

from pydantic import Field, create_model

from flexsignal.core.registry import registered_steps
from flexsignal.spec.models import SignalSpec

SCHEMA_PATH = Path(__file__).parent / "schemas" / "signal_spec.schema.json"
"""Where the committed JSON Schema lives."""


def build_json_schema() -> dict[str, Any]:
    """Return the JSON Schema of a SignalSpec, with every registered step kind.

    ``steps`` becomes a union discriminated by ``kind`` over the registered step
    classes, so editors and UIs can validate and autocomplete each step's options.
    """
    step_classes = tuple(registered_steps().values())
    # Union[...] (not X | Y): the members are only known at run time.
    any_step = Union[step_classes]  # noqa: UP007
    step_union = Annotated[any_step, Field(discriminator="kind")]
    model = create_model(
        "SignalSpec",
        __base__=SignalSpec,
        __doc__=SignalSpec.__doc__,
        steps=(
            tuple[step_union, ...],
            Field((), description="The steps, applied in order."),
        ),
    )
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def schema_text() -> str:
    """Return the JSON Schema as the exact text that is committed."""
    return json.dumps(build_json_schema(), indent=2, sort_keys=True) + "\n"


def export_json_schema(path: str | Path = SCHEMA_PATH) -> Path:
    """Write the JSON Schema to ``path`` (default: the committed location).

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema_text(), encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":  # pragma: no cover
    print(f"wrote {export_json_schema()}")
