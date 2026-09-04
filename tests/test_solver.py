"""The exhaustive solver must agree with brute-force world enumeration everywhere."""

import json
import random
from pathlib import Path

import pytest

from socialmaze.hrd.io import load_scenarios
from socialmaze.hrd.rules import CRIMINAL, INVESTIGATOR, LUNATIC, RUMORMONGER, GameConfig, Statement
from socialmaze.hrd.scenario import Answer, Scenario
from socialmaze.hrd.simulate import RandomPolicy, StrategicPolicy, simulate_game
from socialmaze.hrd.solver import (
    Hypothesis,
    Perspective,
    Solution,
    analyze,
    exclusion_reason,
    solve,
    solve_perspective,
    solve_prefixes,
)

from .brute_force import brute_force_solve

LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "hidden_role_deduction" / "data"

CONFIG_KEYS = [
    (6, "original"), (6, "rumormonger"), (6, "lunatic"), (6, "full"),
    (10, "full"), (10, "lunatic"), (10, "rumormonger"),
]


def rounds_from(spec):
    """``[[(target, claim), ...], ...]`` with speakers 1..n in order -> rounds of Statements."""
    return [[Statement(i, t, c) for i, (t, c) in enumerate(rnd, start=1)] for rnd in spec]


def scenario_from(config, roles, spec, scenario_id="hand"):
    rounds = rounds_from(spec)
    criminal = next(p for p, r in roles.items() if r == CRIMINAL)
    sc = Scenario(scenario_id, config, "Investigator" if roles[1] in (INVESTIGATOR, RUMORMONGER) else "Criminal",
                  rounds, Answer(criminal, roles[1]), roles=dict(roles))
    sc.validate()
    return sc


# Regression (a): the hypothesis {3, 5, 6} leaves two Criminal candidates. The legacy
# checker ignored such hypotheses and reported the game as uniquely solvable.
TWO_CANDIDATES = scenario_from(
    GameConfig.create(6, "full"),
    {1: LUNATIC, 2: INVESTIGATOR, 3: INVESTIGATOR, 4: CRIMINAL, 5: INVESTIGATOR, 6: RUMORMONGER},
    [
        [(2, "is"), (1, "is_not"), (1, "is_not"), (1, "is_not"), (3, "is_not"), (3, "is_not")],
        [(3, "is"), (5, "is_not"), (6, "is_not"), (2, "is_not"), (6, "is_not"), (5, "is_not")],
        [(4, "is"), (3, "is_not"), (6, "is_not"), (6, "is_not"), (3, "is_not"), (3, "is_not")],
    ],
    "two-candidates",
)

# Regression (b): lunatic variant, Player 1 told "Investigator". There is no Rumormonger
# in this game, so the only perspective is Investigator; the legacy checker also ran a
# Rumormonger perspective (consistent here) and dropped the game as "Unknown".
LUNATIC_INVESTIGATOR = scenario_from(
    GameConfig.create(6, "lunatic"),
    {1: INVESTIGATOR, 2: INVESTIGATOR, 3: INVESTIGATOR, 4: CRIMINAL, 5: LUNATIC, 6: INVESTIGATOR},
    [
        [(4, "is"), (4, "is"), (6, "is_not"), (2, "is_not"), (3, "is_not"), (4, "is")],
        [(2, "is_not"), (5, "is_not"), (4, "is"), (3, "is_not"), (2, "is_not"), (1, "is_not")],
        [(3, "is_not"), (1, "is_not"), (5, "is_not"), (6, "is_not"), (6, "is_not"), (2, "is_not")],
    ],
    "lunatic-investigator",
)


def assert_matches_brute_force(config, displayed_role, statements):
    sol = solve(config, displayed_role, statements)
    ref = brute_force_solve(config, displayed_role, statements)
    assert sol.possible_criminals == ref["possible_criminals"]
    assert sol.possible_p1_roles == ref["possible_p1_roles"]
    assert sol.unique == ref["unique"]
    assert sol.unique == (sol.criminal is not None and sol.player1_role is not None)
    if sol.unique:
        assert [sol.criminal] == ref["possible_criminals"]
        assert [sol.player1_role] == ref["possible_p1_roles"]
    return sol


@pytest.mark.parametrize("n,variant", CONFIG_KEYS)
def test_solver_matches_brute_force_on_random_games(n, variant):
    cfg = GameConfig.create(n, variant)
    rng = random.Random(n * 10 + len(variant))
    policies = [RandomPolicy(), StrategicPolicy()]
    displayed = set()
    verdicts = {True: 0, False: 0}
    for i in range(150):
        role = rng.choice(cfg.roles_present())
        sc = simulate_game(cfg, role, policies[i % 2], rng)
        displayed.add(sc.displayed_role)
        for t in range(1, cfg.num_rounds + 1):
            sol = assert_matches_brute_force(cfg, sc.displayed_role, sc.statements_through(t))
            # The true world is always among the consistent ones.
            assert sc.criminal in sol.possible_criminals
            assert sc.player1_role in sol.possible_p1_roles
            verdicts[sol.unique] += 1
    assert displayed == {INVESTIGATOR, CRIMINAL}
    assert verdicts[True] > 0 and verdicts[False] > 0


@pytest.mark.parametrize("n,variant", CONFIG_KEYS)
def test_prefixes_are_monotone(n, variant):
    cfg = GameConfig.create(n, variant)
    rng = random.Random(n + len(variant))
    for _ in range(60):
        sc = simulate_game(cfg, rng.choice(cfg.roles_present()), RandomPolicy(), rng)
        solutions = solve_prefixes(cfg, sc.displayed_role, sc.rounds)
        assert len(solutions) == cfg.num_rounds
        assert solutions[-1].to_dict() == solve(cfg, sc.displayed_role, sc.all_statements()).to_dict()
        for earlier, later in zip(solutions, solutions[1:]):
            assert set(later.possible_criminals) <= set(earlier.possible_criminals)
            assert set(later.possible_p1_roles) <= set(earlier.possible_p1_roles)
            if earlier.unique:
                assert later.unique
                assert (later.criminal, later.player1_role) == (earlier.criminal, earlier.player1_role)


@pytest.mark.parametrize("name", sorted(p.name for p in LEGACY_DIR.glob("*.json")))
def test_legacy_samples_are_unique_with_the_stored_answer(name):
    scenarios = load_scenarios(LEGACY_DIR / name)
    assert len(scenarios) == 10
    for sc in scenarios:
        sol = assert_matches_brute_force(sc.config, sc.displayed_role, sc.all_statements())
        assert sol.unique
        assert (sol.criminal, sol.player1_role) == (sc.criminal, sc.player1_role)
        analysis = analyze(sc)
        assert analysis["unique"] and analysis["criminal"] == sc.criminal
        assert analysis["player1_role"] == sc.player1_role
        assert analysis["solvable_after_round"] in (1, 2, 3)
        assert analysis["possible_criminals_by_round"][-1] == [sc.criminal]


def test_hypothesis_with_two_candidates_makes_the_game_ambiguous():
    sc = TWO_CANDIDATES
    sol = assert_matches_brute_force(sc.config, sc.displayed_role, sc.all_statements())
    assert not sol.unique
    assert sol.possible_criminals == [2, 4]
    assert sol.possible_p1_roles == [LUNATIC]
    criminal_case, lunatic_case = sol.perspectives
    assert criminal_case.p1_role == CRIMINAL and not criminal_case.possible
    assert set(criminal_case.excluded) == {2, 3, 4}
    assert criminal_case.eligible == [5, 6] and not criminal_case.enough_eligible
    assert lunatic_case.p1_role == LUNATIC and lunatic_case.possible
    assert lunatic_case.k == 3 and lunatic_case.excluded == {}
    by_combo = {h.investigators: h for h in lunatic_case.hypotheses}
    assert len(by_combo) == 10
    assert by_combo[(3, 5, 6)].consistent and by_combo[(3, 5, 6)].candidates == {2, 4}
    assert by_combo[(2, 3, 6)].consistent and by_combo[(2, 3, 6)].candidates == {4}
    assert not by_combo[(2, 3, 4)].consistent and by_combo[(2, 3, 4)].candidates == frozenset()
    assert by_combo[(2, 3, 4)].first_contradiction.statement == Statement(3, 6, "is_not")
    assert sol.num_consistent_hypotheses == 4
    analysis = analyze(sc)
    assert analysis["unique"] is False and analysis["criminal"] is None
    assert analysis["solvable_after_round"] is None
    assert analysis["possible_criminals_by_round"][-1] == [2, 4]
    assert analysis["p1_cross_checked_by_investigator"] is True


def test_lunatic_variant_told_investigator_has_one_perspective():
    sc = LUNATIC_INVESTIGATOR
    sol = assert_matches_brute_force(sc.config, sc.displayed_role, sc.all_statements())
    assert [p.p1_role for p in sol.perspectives] == [INVESTIGATOR]
    assert sol.unique and sol.criminal == 4 and sol.player1_role == INVESTIGATOR
    case = sol.perspectives[0]
    assert case.k == 3
    assert case.excluded == {4: "Player 4 is the Criminal by my own statement, so Player 4 cannot be an Investigator."}
    assert case.eligible == [2, 3, 5, 6]
    assert case.p1_statement_notes[0] == "I said that Player 4 is the criminal, so Player 4 is the Criminal."
    assert case.p1_contradiction is None
    for h in case.hypotheses:
        assert h.truthful == (1,) + h.investigators
        assert h.steps[0].statement == Statement(1, 4, "is")
        assert h.candidates == {4}
    analysis = analyze(sc)
    assert analysis == {
        "unique": True,
        "criminal": 4,
        "player1_role": INVESTIGATOR,
        "possible_criminals": [4],
        "possible_p1_roles": [INVESTIGATOR],
        "num_consistent_hypotheses": 4,
        "solvable_after_round": 1,
        "possible_criminals_by_round": [[4], [4], [4]],
        "p1_cross_checked_by_investigator": True,
    }


def test_cross_check_is_none_without_complete_roles():
    sc = LUNATIC_INVESTIGATOR
    partial = Scenario(sc.id, sc.config, sc.displayed_role, sc.rounds, sc.answer, roles={1: INVESTIGATOR, 4: CRIMINAL})
    assert analyze(partial)["p1_cross_checked_by_investigator"] is None


def test_quick_exclusion_reasons():
    cfg = GameConfig.create(6, "full")
    statements = [
        Statement(2, 1, "is"),          # accuses Player 1
        Statement(3, 1, "is_not"),      # clears Player 1
        Statement(4, 5, "is"),          # accuses someone else
        Statement(5, 2, "is"), Statement(5, 3, "is"),        # two accusations
        Statement(6, 2, "is"), Statement(6, 2, "is_not"),    # accuses and clears
        Statement(1, 3, "is"),          # Player 1 accuses Player 3
    ]
    investigator = solve_perspective(cfg, INVESTIGATOR, statements)
    assert set(investigator.excluded) == {2, 3, 5, 6}
    assert "said that I am the criminal" in investigator.excluded[2]
    assert "by my own statement" in investigator.excluded[3]
    assert "Player 2 is the criminal and also that Player 3 is the criminal" in investigator.excluded[5]
    assert "Player 2 is the criminal and also that Player 2 is not the criminal" in investigator.excluded[6]
    rumormonger = solve_perspective(cfg, RUMORMONGER, statements)
    assert set(rumormonger.excluded) == {2, 5, 6}
    criminal = solve_perspective(cfg, CRIMINAL, statements)
    assert set(criminal.excluded) == {3, 4, 5, 6}
    assert "I am not the criminal, but I am the Criminal" in criminal.excluded[3]
    assert "Player 5 is the criminal, but I am the Criminal" in criminal.excluded[4]
    lunatic = solve_perspective(cfg, LUNATIC, statements)
    assert set(lunatic.excluded) == {2, 5, 6}
    assert exclusion_reason(2, [Statement(2, 3, "is_not")], INVESTIGATOR) is None


def test_player_one_contradiction_rules_out_the_investigator_case():
    cfg = GameConfig.create(6, "rumormonger")
    statements = [Statement(1, 2, "is"), Statement(1, 3, "is"), Statement(2, 4, "is_not")]
    sol = solve(cfg, INVESTIGATOR, statements)
    investigator, rumormonger = sol.perspectives
    assert investigator.p1_contradiction == (
        "I said that Player 2 is the criminal and also that Player 3 is the criminal, "
        "which cannot both be true, so I cannot be an Investigator."
    )
    assert not investigator.possible and rumormonger.possible
    assert rumormonger.p1_contradiction is None
    assert sol.possible_p1_roles == [RUMORMONGER]


def test_solution_to_dict_is_json_serialisable():
    sc = LUNATIC_INVESTIGATOR
    d = solve(sc.config, sc.displayed_role, sc.all_statements()).to_dict()
    json.dumps(d)
    assert d["unique"] and d["criminal"] == 4 and d["config"] == sc.config.to_dict()
    assert d["perspectives"][0]["excluded"] == {"4": "Player 4 is the Criminal by my own statement, so Player 4 cannot be an Investigator."}
    assert d["perspectives"][0]["num_hypotheses"] == 4


def test_solve_rejects_statements_about_unknown_players():
    with pytest.raises(ValueError):
        solve(GameConfig.create(6, "full"), INVESTIGATOR, [Statement(2, 7, "is")])
