"""Repo-wide pytest configuration: tier-marker enforcement (mirrors flex-pse)."""

import os

import pytest

TIER_MARKERS = {"unit", "component", "integration"}


def _make_idaes_dirs():
    """Create IDAES's data folders before any pytest worker imports flex-pse.

    Importing IDAES (through flex-pse) creates its data folder by checking that
    it doesn't exist and then making it. Under ``pytest -n auto`` several worker
    processes do that at once, and the losers fail with ``FileExistsError``.
    Making the folders here, with ``exist_ok``, takes the race away. The paths
    follow ``idaes.config.get_data_directory``; nothing is imported.
    """
    if "IDAES_DATA" in os.environ:
        data = os.environ["IDAES_DATA"]
    elif os.name == "nt":
        data = os.path.join(os.environ.get("LOCALAPPDATA", ""), "idaes")
    else:
        data = os.path.join(os.environ.get("HOME", ""), ".idaes")
    if not os.path.isdir(os.path.dirname(data)):
        return  # IDAES itself then runs without a data folder
    for folder in (data, os.path.join(data, "bin"), os.path.join(data, "testing")):
        os.makedirs(folder, exist_ok=True)


_make_idaes_dirs()


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
