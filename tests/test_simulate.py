"""Simulation: role assignment, statement generation and the two targeting policies."""

import random
from collections import Counter

import pytest

from socialmaze.hrd.rules import (
    CRIMINAL,
    DISPLAYED_ROLE,
    INVESTIGATOR,
    LUNATIC,
    RUMORMONGER,
    VARIANTS,
    GameConfig,
    Statement,
)
from socialmaze.hrd.simulate import (
    POLICIES,
    RandomPolicy,
    StrategicPolicy,
    assign_roles,
    make_policy,
    simulate_game,
    suspicion_weight,
)

CONFIGS = [GameConfig.create(n, v) for n in (6, 10) for v in VARIANTS]
CONFIG_IDS = [f"n{c.num_players}-{c.variant}" for c in CONFIGS]


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_assign_roles_honours_counts_and_player_one(config):
    rng = random.Random(0)
    for role in config.roles_present():
        roles = assign_roles(config, role, rng)
        assert roles[1] == role
        assert sorted(roles) == list(config.players)
        assert Counter(roles.values()) == {r: c for r, c in config.role_counts.items() if c}


def test_assign_roles_rejects_absent_roles():
    with pytest.raises(ValueError):
        assign_roles(GameConfig.create(6, "original"), RUMORMONGER, random.Random(0))
    with pytest.raises(ValueError):
        assign_roles(GameConfig.create(6, "rumormonger"), LUNATIC, random.Random(0))


@pytest.mark.parametrize("policy_name", sorted(POLICIES))
@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_simulated_games_follow_the_rules(config, policy_name):
    policy = make_policy(policy_name)
    rng = random.Random(1)
    present = config.roles_present()
    for i in range(16):
        role = present[i % len(present)]
        sc = simulate_game(config, role, policy, rng, scenario_id=f"game-{i}")
        sc.validate()
        assert sc.id == f"game-{i}"
        assert sc.config == config
        assert sc.player1_role == role and sc.roles[1] == role
        assert sc.displayed_role == DISPLAYED_ROLE[role]
        assert sc.roles[sc.criminal] == CRIMINAL
        assert sc.has_complete_roles()
        assert len(sc.rounds) == config.num_rounds
        for rnd in sc.rounds:
            assert [s.speaker for s in rnd] == list(config.players)
            for s in rnd:
                assert s.target != s.speaker
                assert 1 <= s.target <= config.num_players
        for s in sc.all_statements():
            if sc.roles[s.speaker] == INVESTIGATOR:
                assert s.holds_for(sc.criminal)
        assert sc.meta == {"targeting": policy_name}
        assert sc.solution is None and sc.reasoning is None


@pytest.mark.parametrize("policy_name", sorted(POLICIES))
def test_same_seed_same_game_different_seed_different_game(policy_name):
    cfg = GameConfig.create(6, "full")
    a = simulate_game(cfg, LUNATIC, make_policy(policy_name), random.Random(3))
    b = simulate_game(cfg, LUNATIC, make_policy(policy_name), random.Random(3))
    c = simulate_game(cfg, LUNATIC, make_policy(policy_name), random.Random(4))
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()


def test_random_policy_reproduces_the_release_distribution():
    cfg = GameConfig.create(6, "full")
    rng = random.Random(5)
    policy = RandomPolicy()
    targets = {p: Counter() for p in cfg.players}
    claims = Counter()
    for _ in range(1000):
        sc = simulate_game(cfg, RUMORMONGER, policy, rng)
        for s in sc.all_statements():
            if sc.roles[s.speaker] != INVESTIGATOR:
                targets[s.speaker][s.target] += 1
                claims[s.claim] += 1
    total = sum(claims.values())
    assert abs(claims["is"] / total - 0.5) < 0.03
    # Every speaker targets each of the other players about equally often.
    for speaker, counts in targets.items():
        assert speaker not in counts
        n = sum(counts.values())
        assert set(counts) == set(cfg.players) - {speaker}
        for count in counts.values():
            assert abs(count / n - 0.2) < 0.05


def test_strategic_policy_targets_suspects_and_accusers():
    cfg = GameConfig.create(6, "full")
    roles = {1: INVESTIGATOR, 2: INVESTIGATOR, 3: INVESTIGATOR, 4: CRIMINAL, 5: RUMORMONGER, 6: LUNATIC}
    # Players 4, 5 and 6 all accused Player 3; Player 3 accused Player 4.
    history = [
        Statement(3, 4, "is"), Statement(4, 3, "is"), Statement(5, 3, "is"), Statement(6, 3, "is"),
    ]
    assert suspicion_weight(3, history) == 7.0
    assert suspicion_weight(4, history) == 3.0
    assert suspicion_weight(2, history) == 1.0
    policy = StrategicPolicy()
    rng = random.Random(0)
    investigator_targets = Counter()
    criminal_targets = Counter()
    criminal_claims_about_3 = Counter()
    for _ in range(2000):
        s = policy.statement(2, roles, 4, history, rng)
        assert s.speaker == 2 and s.target != 2
        assert s.holds_for(4)
        investigator_targets[s.target] += 1
        c = policy.statement(4, roles, 4, history, rng)
        assert c.speaker == 4 and c.target != 4
        criminal_targets[c.target] += 1
        if c.target == 3:
            criminal_claims_about_3[c.claim] += 1
    # Investigator: Player 3 has weight 7 of 13 in total.
    assert abs(investigator_targets[3] / 2000 - 7 / 13) < 0.05
    # Criminal: Player 3 (an accuser) has weight 4 of 8 and is accused back 80% of the time.
    assert abs(criminal_targets[3] / 2000 - 0.5) < 0.05
    share_is = criminal_claims_about_3["is"] / sum(criminal_claims_about_3.values())
    assert abs(share_is - 0.8) < 0.05


def test_strategic_policy_floors_weights_and_penalises_contradictions():
    history = [Statement(2, 5, "is_not"), Statement(3, 5, "is_not"), Statement(4, 5, "is_not")]
    assert suspicion_weight(5, history) == pytest.approx(0.2)
    contradictory = [Statement(6, 2, "is"), Statement(6, 3, "is")]
    assert suspicion_weight(6, contradictory) == 4.0


def test_make_policy_names():
    assert make_policy("random").name == "random"
    assert make_policy("Strategic").name == "strategic"
    with pytest.raises(ValueError):
        make_policy("clever")
