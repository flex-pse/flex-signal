"""Repo-wide pytest configuration: tier-marker enforcement (mirrors flex-pse)."""

import pytest

TIER_MARKERS = {"unit", "component", "integration"}


def pytest_collection_modifyitems(config, items):
    """Fail collection if any test carries zero or more than one tier marker."""
    errors = []
    for item in items:
        tiers = TIER_MARKERS & {m.name for m in item.iter_markers()}
        if len(tiers) != 1:
            errors.append(
                f"{item.nodeid}: needs exactly one tier marker "
                f"(unit/component/integration), got {sorted(tiers) or 'none'}"
            )
    if errors:
        raise pytest.UsageError("\n".join(errors))
