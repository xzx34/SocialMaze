"""Reasoning chains: structure, ending, determinism and no mention of absent roles."""

from pathlib import Path

import pytest

from socialmaze.hrd.explain import combo_label, explain, players_phrase, role_with_article
from socialmaze.hrd.generate import generate_dataset
from socialmaze.hrd.io import load_scenarios
from socialmaze.hrd.prompts import answer_block
from socialmaze.hrd.rules import CRIMINAL, LUNATIC, RUMORMONGER, VARIANTS, GameConfig
from socialmaze.hrd.solver import solve

from .test_solver import LUNATIC_INVESTIGATOR, TWO_CANDIDATES

LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "hidden_role_deduction" / "data"


def check_explanation(sc, text):
    assert text.endswith(answer_block(sc.criminal, sc.player1_role))
    assert text.count("Final Judgment:") == 1
    if sc.config.num_rumormongers == 0:
        assert RUMORMONGER not in text
    if sc.config.num_lunatics == 0:
        assert LUNATIC not in text
    assert text.startswith(f"I was told that I am {role_with_article(sc.displayed_role)}.")
    perspectives = sc.config.true_roles_for(sc.displayed_role)
    for role in perspectives:
        assert f"I am {role_with_article(role)}" in text
    if len(perspectives) == 1:
        assert "The only case is that I am" in text
    else:
        assert "Case 1: I am" in text and "Case 2: I am" in text
    assert "*" not in text and "#" not in text
    assert explain(sc) == text


@pytest.mark.parametrize("name", sorted(p.name for p in LEGACY_DIR.glob("*.json")))
def test_legacy_samples_explain_to_the_stored_answer(name):
    for sc in load_scenarios(LEGACY_DIR / name):
        check_explanation(sc, explain(sc))


@pytest.mark.parametrize("n,variant", [(6, v) for v in VARIANTS] + [(10, "full"), (10, "rumormonger")])
def test_generated_scenarios_explain_to_their_answer(n, variant):
    scenarios, _ = generate_dataset(GameConfig.create(n, variant), 8, seed=n + len(variant))
    for sc in scenarios:
        check_explanation(sc, sc.reasoning)
        assert explain(sc) == sc.reasoning


def test_every_consistent_combination_is_narrated():
    for sc in load_scenarios(LEGACY_DIR / "hrd_10_all.json"):
        sol = solve(sc.config, sc.displayed_role, sc.all_statements())
        text = explain(sc, sol)
        for p in sol.perspectives:
            if p.p1_role == CRIMINAL or p.p1_contradiction is not None:
                continue
            for h in p.consistent_hypotheses:
                assert f"{combo_label(h)}, the Criminal must be" in text
            for h in p.eliminated_hypotheses:
                combo = "{" + ", ".join(str(x) for x in h.investigators) + "}"
                assert combo in text or f"{combo_label(h)}:" in text
            for reason in p.excluded.values():
                assert reason in text


def test_solution_argument_gives_the_same_text():
    sc = LUNATIC_INVESTIGATOR
    sol = solve(sc.config, sc.displayed_role, sc.all_statements())
    assert explain(sc, sol) == explain(sc)
    text = explain(sc)
    check_explanation(sc, text)
    assert "In this game every player who is told that they are an Investigator really is an Investigator" in text
    assert "Player 4 is the Criminal by my own statement" in text


def test_non_unique_scenario_ends_without_final_judgment():
    text = explain(TWO_CANDIDATES)
    assert "Final Judgment" not in text
    assert text.endswith(
        "The statements do not determine a unique answer: the Criminal could be one of "
        "Players 2 and 4, and I am a Lunatic."
    )
    assert "the case where I am the Criminal is impossible" in text
    assert "If the Investigators are Players 3, 5 and 6, the Criminal must be one of Players 2 and 4." in text
    assert explain(TWO_CANDIDATES) == text


def test_phrases():
    assert players_phrase([4]) == "Player 4"
    assert players_phrase([6, 2]) == "Players 2 and 6"
    assert players_phrase([3, 1, 2]) == "Players 1, 2 and 3"
    assert role_with_article(CRIMINAL) == "the Criminal"
