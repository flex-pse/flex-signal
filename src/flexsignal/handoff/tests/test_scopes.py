"""Time scopes: which time block a model path belongs to (from flex-orchestrator)."""

import pandas as pd
import pytest

from flexsignal.core import SignalConfigError
from flexsignal.handoff import grids_of, scope_of, targets

GRID = pd.date_range("2025-07-08", periods=8, freq="15min")


@pytest.mark.unit
def test_a_path_belongs_to_the_nearest_time_block_above_it():
    scopes = ["", "plant", "plant.unit[1]"]
    assert scope_of("plant.unit[1].x", scopes) == "plant.unit[1]"
    assert scope_of("plant.unit[2].x", scopes) == "plant"
    assert scope_of("other.x", scopes) == ""
    assert scope_of("plantx.y", ["plant"]) is None  # a longer name isn't a child
    assert targets("x", ["a", "b"]) == [("a", "a.x"), ("b", "b.x")]
    assert targets("a.x", ["a", "b"]) == [("a", "a.x")]
    # A * in a time block's index matches any value there; one in the
    # variable's own index is left alone.
    horizons = ["h[2030,1]", "h[2030,7]", "h[2035,1]", "h[2035,7]"]
    assert targets("h[2030,*].v[*,Liq]", horizons) == [
        ("h[2030,1]", "h[2030,1].v[*,Liq]"),
        ("h[2030,7]", "h[2030,7].v[*,Liq]"),
    ]
    assert [s for s, _ in targets("h[*,7].v", horizons)] == ["h[2030,7]", "h[2035,7]"]
    assert len(targets("v[*,Liq]", horizons)) == 4  # under none: every horizon
    with pytest.raises(SignalConfigError, match="matches no time block"):
        targets("h[2040,*].v", horizons)
    assert targets("u[*].x", ["a"]) == [("a", "a.u[*].x")]  # not a time block
    assert grids_of(None) == {"": None}
    assert list(grids_of(GRID)) == [""]
    two = {"plant_a": GRID, "plant_b": GRID}
    assert grids_of(two) == two
    with pytest.raises(SignalConfigError, match="time_grid hook"):
        grids_of({})
