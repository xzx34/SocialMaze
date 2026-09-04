import pytest

from socialmaze.hrd.rules import (
    CRIMINAL,
    INVESTIGATOR,
    LUNATIC,
    RUMORMONGER,
    GameConfig,
    Statement,
    default_special_counts,
    normalize_claim,
    normalize_variant,
    variant_from_counts,
)

RELEASED_COUNTS = {
    (6, "original"): {INVESTIGATOR: 5, CRIMINAL: 1, RUMORMONGER: 0, LUNATIC: 0},
    (6, "rumormonger"): {INVESTIGATOR: 4, CRIMINAL: 1, RUMORMONGER: 1, LUNATIC: 0},
    (6, "lunatic"): {INVESTIGATOR: 4, CRIMINAL: 1, RUMORMONGER: 0, LUNATIC: 1},
    (6, "full"): {INVESTIGATOR: 3, CRIMINAL: 1, RUMORMONGER: 1, LUNATIC: 1},
    (10, "original"): {INVESTIGATOR: 9, CRIMINAL: 1, RUMORMONGER: 0, LUNATIC: 0},
    (10, "rumormonger"): {INVESTIGATOR: 7, CRIMINAL: 1, RUMORMONGER: 2, LUNATIC: 0},
    (10, "lunatic"): {INVESTIGATOR: 7, CRIMINAL: 1, RUMORMONGER: 0, LUNATIC: 2},
    (10, "full"): {INVESTIGATOR: 5, CRIMINAL: 1, RUMORMONGER: 2, LUNATIC: 2},
}


@pytest.mark.parametrize("key,counts", RELEASED_COUNTS.items())
def test_default_counts_match_released_configurations(key, counts):
    n, variant = key
    cfg = GameConfig.create(n, variant)
    assert cfg.role_counts == counts
    assert cfg.variant == variant
    assert cfg.num_players == n


def test_variant_aliases_and_inference():
    assert normalize_variant("all") == "full"
    assert normalize_variant("Full") == "full"
    assert GameConfig.create(6, "all").variant == "full"
    assert variant_from_counts(0, 0) == "original"
    assert variant_from_counts(2, 0) == "rumormonger"
    assert variant_from_counts(0, 1) == "lunatic"
    assert variant_from_counts(1, 1) == "full"
    assert GameConfig.create(8, num_rumormongers=0, num_lunatics=2).variant == "lunatic"
    with pytest.raises(ValueError):
        normalize_variant("hard")


def test_inconsistent_variant_and_counts_rejected():
    with pytest.raises(ValueError):
        GameConfig.create(6, "original", num_rumormongers=1)
    with pytest.raises(ValueError):
        GameConfig.create(6, "lunatic", num_rumormongers=1, num_lunatics=1)


def test_at_least_one_investigator():
    with pytest.raises(ValueError):
        GameConfig(num_players=4, num_rumormongers=2, num_lunatics=1)
    GameConfig(num_players=5, num_rumormongers=2, num_lunatics=1)


def test_true_roles_for_displayed_role_only_lists_present_roles():
    assert GameConfig.create(6, "original").true_roles_for(INVESTIGATOR) == (INVESTIGATOR,)
    assert GameConfig.create(6, "original").true_roles_for(CRIMINAL) == (CRIMINAL,)
    assert GameConfig.create(6, "rumormonger").true_roles_for(INVESTIGATOR) == (INVESTIGATOR, RUMORMONGER)
    assert GameConfig.create(6, "rumormonger").true_roles_for(CRIMINAL) == (CRIMINAL,)
    assert GameConfig.create(6, "lunatic").true_roles_for(INVESTIGATOR) == (INVESTIGATOR,)
    assert GameConfig.create(6, "lunatic").true_roles_for(CRIMINAL) == (CRIMINAL, LUNATIC)
    assert GameConfig.create(6, "full").true_roles_for(CRIMINAL) == (CRIMINAL, LUNATIC)


def test_default_special_counts_scale_with_players():
    assert default_special_counts(6, "full") == (1, 1)
    assert default_special_counts(10, "full") == (2, 2)
    assert default_special_counts(4, "full") == (1, 1)
    assert default_special_counts(15, "rumormonger") == (3, 0)


def test_config_dict_round_trip():
    cfg = GameConfig.create(10, "full", num_rounds=4)
    d = cfg.to_dict()
    assert d["role_counts"][INVESTIGATOR] == 5 and d["num_rounds"] == 4
    assert GameConfig.from_dict(d) == cfg
    assert GameConfig.from_dict({"num_players": 6, "variant": "all"}) == GameConfig.create(6, "full")
    with pytest.raises(ValueError):
        GameConfig.from_dict({"num_players": 7, "role_counts": d["role_counts"]})


def test_statement_render_parse_and_truth():
    s = Statement(2, 5, "is not")
    assert s.claim == "is_not" and not s.accuses
    assert s.render() == "Player 2 says Player 5 is not the criminal."
    assert Statement.parse(s.render()) == s
    assert Statement.parse("Player 3 says Player 1 is the criminal.") == Statement(3, 1, "is")
    assert Statement.parse("nothing here") is None
    assert s.holds_for(criminal=4) is True
    assert s.holds_for(criminal=5) is False
    assert Statement(2, 5, "is").holds_for(5) is True
    assert normalize_claim("Is Not") == "is_not"
    with pytest.raises(ValueError):
        Statement(2, 2, "is")
    with pytest.raises(ValueError):
        Statement(2, 3, "maybe")


def test_statement_parse_all_preserves_order():
    text = "Round 1 statements:\nPlayer 1 says Player 6 is the criminal.\nPlayer 2 says Player 3 is not the criminal.\n"
    assert Statement.parse_all(text) == [Statement(1, 6, "is"), Statement(2, 3, "is_not")]
