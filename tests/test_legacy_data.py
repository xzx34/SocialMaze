"""The released sample scenarios must all be uniquely solvable by exhaustive enumeration."""

from pathlib import Path

import pytest

from socialmaze.hrd.io import load_scenarios

from .brute_force import brute_force_solve

LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "hidden_role_deduction" / "data"


@pytest.mark.parametrize("name", sorted(p.name for p in LEGACY_DIR.glob("*.json")))
def test_released_samples_are_unique_by_brute_force(name):
    for sc in load_scenarios(LEGACY_DIR / name):
        result = brute_force_solve(sc.config, sc.displayed_role, sc.all_statements())
        assert result["unique"], sc.id
        assert result["possible_criminals"] == [sc.criminal]
        assert result["possible_p1_roles"] == [sc.player1_role]
