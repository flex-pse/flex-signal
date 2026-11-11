"""The docs stay correct: their code runs and the generated tables match the code.

Each page's ``python`` code blocks run in order in a fresh interpreter and a
temporary directory, as a reader would run them. Blocks written as
``{code-block} python`` are illustrative only and are not run. The marimo
tutorials in ``tutorials/`` run as scripts. These tests are skipped when the docs
or tutorials aren't present, e.g. from an installed wheel.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
PAGES = (
    sorted([*DOCS.glob("explanation/*.md"), *DOCS.glob("how_to/*.md")])
    if DOCS.exists()
    else []
)
PAGES_WITH_CODE = [p for p in PAGES if FENCE.search(p.read_text(encoding="utf-8"))]


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.integration
@pytest.mark.skipif(not PAGES_WITH_CODE, reason="docs not present")
@pytest.mark.parametrize(
    "page", PAGES_WITH_CODE, ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_page_code_runs(page, tmp_path):
    blocks = FENCE.findall(page.read_text(encoding="utf-8"))
    result = _run("\n\n".join(blocks), tmp_path)
    assert result.returncode == 0, f"{page.name} failed:\n{result.stderr[-3000:]}"


@pytest.mark.integration
@pytest.mark.skipif(not DOCS.exists(), reason="docs not present")
@pytest.mark.parametrize(
    "script",
    sorted(DOCS.glob("examples/*.py")) if DOCS.exists() else [],
    ids=lambda p: p.stem,
)
def test_example_script_runs(script, tmp_path):
    result = _run(script.read_text(encoding="utf-8"), tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "removed exactly: True" in result.stdout


TUTORIALS = sorted((REPO / "tutorials").glob("[0-9][0-9]_*.py"))


@pytest.mark.integration
@pytest.mark.skipif(not TUTORIALS, reason="tutorials not present")
@pytest.mark.parametrize("tutorial", TUTORIALS, ids=lambda p: p.stem)
def test_tutorial_runs_as_a_script(tutorial, tmp_path):
    # Each tutorial ends with a cell of asserts about its own results, so running
    # it checks what it teaches, not only that it doesn't crash.
    pytest.importorskip("marimo", reason="needs the tutorials extra")
    pytest.importorskip("matplotlib", reason="needs the tutorials extra")
    result = subprocess.run(
        [sys.executable, str(tutorial)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.unit
@pytest.mark.skipif(not TUTORIALS, reason="tutorials not present")
def test_tutorials_are_numbered_in_order():
    numbers = [int(p.name[:2]) for p in TUTORIALS]
    assert numbers == list(range(1, len(numbers) + 1))


@pytest.mark.unit
@pytest.mark.skipif(not DOCS.exists(), reason="docs not present")
def test_generated_tables_match_the_registries(monkeypatch):
    pytest.importorskip("docutils")
    monkeypatch.syspath_prepend(str(DOCS / "_ext"))
    import flexsignaldoc

    from flexsignal.backends import registered_backends
    from flexsignal.core import FLAG_DESCRIPTIONS, registered_steps
    from flexsignal.steps.clean import Hampel

    steps = flexsignaldoc.collect_step_rows()
    assert [row[0] for row in steps] == [f"``{k}``" for k in registered_steps()]
    stateful = {row[0]: row[3] for row in steps}
    assert stateful["``tukey``"] == "yes" and stateful["``hampel``"] == "no"
    fields = {row[0]: row for row in flexsignaldoc.collect_spec_rows(Hampel)}
    assert fields["``window``"][2] == "required"
    assert fields["``n_sigma``"][2] == "``3.0``"
    assert len(flexsignaldoc.collect_flag_rows()) == len(FLAG_DESCRIPTIONS)
    assert len(flexsignaldoc.collect_backend_rows()) == len(registered_backends())
