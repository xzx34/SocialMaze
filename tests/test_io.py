import json
from pathlib import Path

import pytest

from socialmaze.hrd.io import (
    from_hf_row,
    load_meta,
    load_scenarios,
    parse_answer_text,
    parse_config_from_system_prompt,
    save_scenarios,
    to_hf_row,
)
from socialmaze.hrd.prompts import system_prompt
from socialmaze.hrd.rules import CRIMINAL, DISPLAYED_ROLES, INVESTIGATOR, LUNATIC, VARIANTS, GameConfig
from socialmaze.hrd.scenario import Scenario

REPO = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO / "archive" / "hidden_role_deduction" / "data"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("n", [6, 10])
@pytest.mark.parametrize("dtype", ["original", "rumormonger", "lunatic", "all"])
def test_legacy_samples_load_and_validate(n, dtype):
    scenarios = load_scenarios(LEGACY_DIR / f"hrd_{n}_{dtype}.json")
    assert len(scenarios) == 10
    for sc in scenarios:
        sc.validate()
        assert sc.num_players == n
        assert sc.variant == ("full" if dtype == "all" else dtype)
        assert sc.has_complete_roles()
        assert sc.meta["source"] == "legacy"
        assert sc.reasoning


def test_hf_row_fixture_parses():
    row = json.loads((FIXTURES / "hf_row_easy.json").read_text())
    sc = from_hf_row(row, index=3, id_prefix="hf-easy")
    sc.validate()
    assert sc.id == "hf-easy-000003"
    assert sc.config == GameConfig.create(6, "full")
    assert sc.displayed_role == CRIMINAL
    assert sc.player1_role == LUNATIC and sc.criminal == 4
    assert sc.roles == {1: LUNATIC, 4: CRIMINAL}
    assert not sc.has_complete_roles()
    assert [len(r) for r in sc.rounds] == [6, 6, 6]
    assert sc.rounds[0][0].render() == "Player 1 says Player 6 is the criminal."
    assert sc.reasoning.startswith("I'm told I'm a Criminal")


def test_hf_export_round_trip():
    scenarios = load_scenarios(LEGACY_DIR / "hrd_6_all.json")
    for sc in scenarios:
        row = to_hf_row(sc)
        assert set(row) >= {"task", "system_prompt", "prompt", "answer", "reasoning_process", "round 1", "round 2", "round 3", "id"}
        assert row["answer"] == f"Final Criminal Is Player {sc.criminal}.\nMy Role Is {sc.player1_role}."
        assert row["round 2"].startswith("Round 2 statements:\n")
        back = from_hf_row(row)
        back.validate()
        assert back.id == sc.id
        assert back.config == sc.config
        assert back.displayed_role == sc.displayed_role
        assert back.rounds == sc.rounds
        assert back.answer == sc.answer


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("displayed", DISPLAYED_ROLES)
@pytest.mark.parametrize("n", [6, 10])
def test_current_system_prompt_is_parseable(variant, displayed, n):
    cfg = GameConfig.create(n, variant)
    parsed_cfg, told = parse_config_from_system_prompt(system_prompt(cfg, displayed))
    assert parsed_cfg == cfg
    assert told == displayed


def test_parse_answer_text():
    a = parse_answer_text("Final Criminal Is Player 4.\nMy Role Is Lunatic.")
    assert (a.criminal, a.player1_role) == (4, LUNATIC)
    with pytest.raises(ValueError):
        parse_answer_text("no answer")


def test_jsonl_round_trip_with_meta(tmp_path):
    scenarios = load_scenarios(LEGACY_DIR / "hrd_10_all.json")
    out = tmp_path / "sub" / "hrd_n10_full.jsonl"
    save_scenarios(out, scenarios, meta={"note": "test", "count": len(scenarios)})
    assert (tmp_path / "sub" / "hrd_n10_full.meta.json").exists()
    assert load_meta(out) == {"note": "test", "count": 10}
    back = load_scenarios(out)
    assert [s.to_dict() for s in back] == [s.to_dict() for s in scenarios]
    assert load_scenarios(out, limit=3) == back[:3]


def test_load_scenarios_accepts_hf_rows_in_json_and_jsonl(tmp_path):
    scenarios = load_scenarios(LEGACY_DIR / "hrd_6_lunatic.json", limit=4)
    rows = [to_hf_row(s) for s in scenarios]
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for name in ("rows.json", "rows.jsonl"):
        back = load_scenarios(tmp_path / name)
        assert [s.id for s in back] == [s.id for s in scenarios]
        assert all(b.rounds == s.rounds for b, s in zip(back, scenarios))
    with pytest.raises(ValueError):
        load_scenarios(tmp_path / "rows.txt")


def test_scenario_validation_catches_inconsistencies():
    sc = load_scenarios(LEGACY_DIR / "hrd_6_all.json", limit=1)[0]
    d = sc.to_dict()
    d["answer"]["criminal"] = 1
    with pytest.raises(ValueError):
        Scenario.from_dict(d).validate()
    d = sc.to_dict()
    d["displayed_role"] = CRIMINAL if sc.displayed_role == INVESTIGATOR else INVESTIGATOR
    with pytest.raises(ValueError):
        Scenario.from_dict(d).validate()
