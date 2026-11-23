"""flex-pse: a model's time grid, and fixing model variables from data.

- :func:`time_grid`: the time points of a flowsheet spec or a built model, one
  axis per time block.
- :func:`apply_values`: fix variables from values, on a flowsheet spec (a new
  spec, to build) or on a built model (in place), with the same result.

Values come from :func:`flexsignal.handoff.to_grid`, via
:meth:`~flexsignal.handoff.GridValues.to_overrides`.

flex-pse imports flex-signal, never the reverse at import time: this is the only
module allowed to touch flex-pse, pyomo or idaes, and it imports them inside
functions, so ``import flexsignal`` stays light. flex-pse isn't on PyPI
yet: install its latest release from its git tag. These
functions came from flex-orchestrator, which now calls them.
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Mapping
from typing import Any

import pandas as pd

from flexsignal.core.exceptions import SignalBackendError, SignalConfigError
from flexsignal.handoff.scopes import scope_of

__all__ = ["apply_values", "time_grid", "time_scopes"]

_DESIGN_FIELDS = frozenset({"base", "years", "months", "capacities"})
_PRICE_FIELDS = frozenset({"energy_prices", "export_prices"})


def _need_flexpse(what: str):
    try:
        import flexops  # noqa: F401
        import pyomo.environ  # noqa: F401
    except ImportError as exc:
        raise SignalBackendError(
            f"{what} needs flex-pse: install it with "
            "`pip install flex-signal[flexpse]` (flex-pse is installed from its git "
            "tag; see the installation guide)."
        ) from exc


def time_grid(spec_or_model: Any) -> pd.DatetimeIndex | dict | None:
    """Return the time points of a flex-pse flowsheet spec or built model.

    For a flowsheet spec, the points ``flexops`` builds the model's time block
    on: from ``start_date`` (inclusive) to ``end_date`` (exclusive) every
    ``time_step``, in naive local time. For a built model, its time blocks'.

    A model with several time blocks has a time axis per **scope**, the block
    that holds each time block: a model with ``m.plant_a.time_block`` and
    ``m.plant_b.time_block`` has scopes ``plant_a`` and ``plant_b``. Design specs
    aren't read here; call this on each horizon's flowsheet spec.

    Args:
        spec_or_model: A flowsheet spec as JSON (a dict), or a built model. Not
            modified.

    Returns:
        The points, for one time block; ``{scope: points}`` for several; None if
        there's no time axis.

    Raises:
        SignalConfigError: If the spec has two time elements, its dates or step
            don't make a whole number of steps, its dates have a time zone, it is
            a design spec, or one block holds two time blocks.
    """
    if not isinstance(spec_or_model, Mapping):
        return _of_model(spec_or_model)
    if _DESIGN_FIELDS <= spec_or_model.keys():
        raise SignalConfigError(
            "time_grid reads flowsheet specs and built models, not design specs; "
            "call it on each horizon's flowsheet spec."
        )
    return _of_spec(spec_or_model)


def time_scopes(model: Any) -> dict[str, Any]:
    """Return ``{scope: block}`` for every block of ``model`` holding a time block.

    The scope is the block's path from ``model`` (``""`` for the model itself).

    Raises:
        SignalConfigError: If one block holds two time blocks.
    """
    _need_flexpse("time_scopes")
    import pyomo.environ as pyo
    from flexops.core.time_block import TimeBlockData

    scopes: dict[str, Any] = {}
    for block in model.component_data_objects(pyo.Block, descend_into=True):
        if not isinstance(block, TimeBlockData):
            continue
        owner = block.parent_block()
        scope = (
            ""
            if owner is model
            else owner.getname(fully_qualified=True, relative_to=model)
        )
        if scope in scopes:
            raise SignalConfigError(
                f"{owner.name!r} holds two time blocks; give each its own block."
            )
        scopes[scope] = owner
    return scopes


def _of_spec(spec: Mapping[str, Any]) -> pd.DatetimeIndex | None:
    found = [e for e in spec.get("elements", []) if e.get("kind") == "time"]
    if not found:
        return None
    if len(found) > 1:
        raise SignalConfigError("The spec has more than one time element.")
    _need_flexpse("time_grid")
    import pyomo.environ as pyo
    from flexops.core.build import parse_quantity
    from pyomo.environ import units as pyunits

    [element] = found
    try:
        start = datetime.datetime.fromisoformat(element["start_date"])
        end = datetime.datetime.fromisoformat(element["end_date"])
        seconds = pyo.value(
            pyunits.convert(parse_quantity(element["time_step"]), pyunits.s)
        )
    except Exception as exc:
        raise SignalConfigError(f"Can't read the spec's time element: {exc}") from exc
    if start.tzinfo is not None or end.tzinfo is not None:
        raise SignalConfigError("The time element's dates must be naive local time.")
    step = pd.Timedelta(seconds=seconds)
    n, remainder = divmod(pd.Timestamp(end) - pd.Timestamp(start), step)
    if remainder != pd.Timedelta(0) or n <= 0:
        raise SignalConfigError(
            f"The time element's horizon {start} -> {end} isn't a whole number of "
            f"{element['time_step']} steps."
        )
    return pd.date_range(start, periods=n, freq=step)


def _of_model(model: Any) -> pd.DatetimeIndex | dict | None:
    _need_flexpse("time_grid")
    import pyomo.environ as pyo
    from flexops.core.time_block import TimeBlockData

    def time_block(block: Any) -> Any:
        return next(
            b
            for b in block.component_data_objects(pyo.Block, descend_into=False)
            if isinstance(b, TimeBlockData)
        )

    grids = {
        scope: pd.DatetimeIndex(time_block(block).datetime_index)
        for scope, block in time_scopes(model).items()
    }
    if len(grids) == 1:
        return next(iter(grids.values()))
    return grids or None


def apply_values(target: Any, values: Mapping[str, Any]) -> Any:
    """Fix variables from values, on a flowsheet spec or on a built model.

    The same values give the same model either way: build the returned spec, or
    apply them to a model built from the original spec.

    - **A spec** (a dict) isn't modified: a copy is returned with one ``fix``
      element per value, named ``"override:<path>"``. A path whose first part
      names a ``costing`` element and whose next part is one of its fields
      (``"costing.energy_prices.electrical"``) edits that field instead, which
      is how a flat price is set.
    - **A built model** is changed in place and returned. A path naming a
      mutable Param (a price passed to flex-pse as a Param, say) sets it; every
      other path is fixed as a ``fix`` element would be, checked against the
      variables' bounds and applied together (``flexops.apply_spec``), all or
      nothing for each time block. In a model with several time blocks, each
      path is fixed against the nearest time block above it. A later apply of a
      point replaces an earlier one.

    Args:
        target: A flowsheet spec as JSON, or a built model.
        values: ``{path: value}``, e.g. from
            :meth:`flexsignal.handoff.GridValues.to_overrides`. The path is a
            Pyomo path from the model root (``"plant.pump.flow_in"``,
            ``"plant.ro[*].status"``). The value is anything a ``fix`` element
            takes: a number in the variable's units, ``{"value", "units"}``, a
            list with one number per time point, or ``{time index: value}``.

    Returns:
        The new spec, or the same model.

    Raises:
        SignalConfigError: If a costing path runs into a value that isn't a dict
            of options; on a model, if a price was built as a constant (flex-pse
            builds a spec's flat ``energy_prices`` that way), a Param value
            doesn't fit its index, or the model has several time blocks and a
            path is under none of them.
        flexcore.exceptions.FlexConfigError: On a model, if a path names no
            variable, or a value doesn't fit it.
    """
    if isinstance(target, Mapping):
        return _spec_with(target, values)
    _need_flexpse("apply_values")
    from flexops import apply_spec

    params = _find_params(target, values)
    settings = [_param_values(params[path], values[path], path) for path in params]
    fixes = {path: value for path, value in values.items() if path not in params}
    scopes = time_scopes(target) if fixes else {}
    if len(scopes) <= 1:
        if fixes:
            apply_spec(target, _spec_with({"elements": []}, fixes)["elements"])
    else:
        grouped: dict[str, dict[str, Any]] = {}
        for path, value in fixes.items():
            scope = scope_of(path, scopes)
            if scope is None:
                raise SignalConfigError(
                    f"Can't fix {path!r}: the model has several time blocks and it "
                    f"is under none of them. Name it from one of {sorted(scopes)}."
                )
            relative = path[len(scope) + 1 :] if scope else path
            grouped.setdefault(scope, {})[relative] = value
        for scope, group in grouped.items():
            apply_spec(scopes[scope], _spec_with({"elements": []}, group)["elements"])
    for setting in settings:
        for data, number in setting:
            data.set_value(number)
    return target


def _spec_with(spec: Mapping[str, Any], values: Mapping[str, Any]) -> dict:
    """Return a copy of ``spec`` with ``values`` as fix elements or costing fields."""
    elements = copy.deepcopy(list(spec["elements"]))
    costing = {e["name"]: e for e in elements if e.get("kind") == "costing"}
    fixes = []
    for path, value in values.items():
        head, _, rest = path.partition(".")
        if head in costing and rest.partition(".")[0] in costing[head]:
            _set_field(costing[head], rest.split("."), value, path)
            continue
        element = {"kind": "fix", "name": f"override:{path}", "variable": path}
        if isinstance(value, Mapping) and set(value) == {"value", "units"}:
            element.update(value)
        else:
            element["value"] = value
        fixes.append(element)
    return {**spec, "elements": [*elements, *fixes]}


def _set_field(target: dict, parts: list[str], value: Any, path: str) -> None:
    """Set ``target[parts[0]][parts[1]]...``, creating missing dicts on the way."""
    *parents, last = parts
    for part in parents:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise SignalConfigError(
                f"Can't set {path!r}: {part!r} is {child!r}, not a dict of options."
            )
        target = child
    target[last] = value


def _find_params(model: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{path: Param}`` for the paths that name a model's Params."""
    import pyomo.environ as pyo

    params = {}
    for path in values:
        found = model.find_component(path)
        if found is not None and found.ctype is pyo.Param:
            if not found.mutable:
                raise SignalConfigError(
                    f"Can't set {path!r}: it is an immutable Param. Declare it with "
                    "mutable=True."
                )
            params[path] = found
            continue
        parts = path.split(".")
        if found is None and len(parts) > 1 and parts[1] in _PRICE_FIELDS:
            raise SignalConfigError(
                f"Can't set {path!r} on this model: flex-pse built the spec's flat "
                "price as a constant. Set it on the spec and build again, or build "
                "the model with the price as a mutable Param and set that."
            )
    return params


def _param_values(param: Any, value: Any, path: str) -> list[tuple[Any, Any]]:
    """Return ``[(Param data, number)]`` for one value, checked before any is set."""
    if isinstance(value, Mapping) and set(value) == {"value", "units"}:
        from flexops.core.build import parse_quantity
        from pyomo.environ import units as pyunits

        given = pyunits.get_units(
            parse_quantity({"value": 1.0, "units": value["units"]})
        )
        wanted = pyunits.get_units(param)
        if wanted is None:
            wanted = pyunits.dimensionless
        factor = pyunits.convert_value(1.0, from_units=given, to_units=wanted)
        value = value["value"]
        if isinstance(value, Mapping):
            value = {k: v * factor for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            value = [v * factor for v in value]
        else:
            value = value * factor
    members = list(param.keys()) if param.is_indexed() else [None]
    if isinstance(value, Mapping):
        missing = [k for k in value if k not in param]
        if missing:
            raise SignalConfigError(f"{path!r} has no index {missing[0]!r}.")
        return [(param[k], v) for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        if len(value) != len(members):
            raise SignalConfigError(
                f"{path!r} has {len(members)} values to set, but {len(value)} were "
                "given."
            )
        return [(param[k], v) for k, v in zip(members, value, strict=True)]
    return [(param[k], value) for k in members]
