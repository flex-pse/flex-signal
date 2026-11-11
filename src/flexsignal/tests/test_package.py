"""Package-level guarantees that hold from M0 onward."""

import importlib
import subprocess
import sys
from importlib.metadata import version

import pytest

import flexsignal

SUBPACKAGES = [
    "flexsignal.core",
    "flexsignal.spec",
    "flexsignal.io",
    "flexsignal.backends",
    "flexsignal.steps",
    "flexsignal.pipeline",
    "flexsignal.eval",
    "flexsignal.api",
    "flexsignal.integrations",
    "flexsignal.integrations.flexpse",
    "flexsignal.integrations.acquirium",
]

# PLAN.md D1/D5: flex-pse workflows import flex-signal, so importing it must
# not drag in flex-pse (circular imports), the pyomo/idaes stack, torch, or
# polars. Optional backends are imported lazily, when used.
FORBIDDEN_ON_IMPORT = [
    "flexcore",
    "flexops",
    "flexparameterize",
    "pyomo",
    "idaes",
    "torch",
    "polars",
    "acquirium",
]


@pytest.mark.unit
def test_version_matches_distribution():
    assert flexsignal.__version__ == version("flex-signal")


@pytest.mark.unit
@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a module docstring"


@pytest.mark.unit
def test_import_loads_no_heavy_or_forbidden_modules():
    # A fresh interpreter, so modules other tests imported can't mask a leak.
    code = (
        "import sys, importlib\n"
        f"for name in {SUBPACKAGES!r}:\n"
        "    importlib.import_module(name)\n"
        f"leaked = sorted(m for m in {FORBIDDEN_ON_IMPORT!r} if m in sys.modules)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = result.stdout.strip()
    assert not leaked, f"importing flexsignal loaded: {leaked}"
