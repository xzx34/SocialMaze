"""Dataset generation: quotas, role mixes, provenance and the CLI round trip."""

import json
from collections import Counter

import pytest

from socialmaze import __version__
from socialmaze.hrd.cli import main
from socialmaze.hrd.generate import allocate_quotas, generate_dataset, parse_role_mix
from socialmaze.hrd.io import from_hf_row, load_meta, load_scenarios, meta_path, to_hf_row, write_jsonl
from socialmaze.hrd.prompts import answer_block
from socialmaze.hrd.rules import CRIMINAL, INVESTIGATOR, LUNATIC, RUMORMONGER, GameConfig
from socialmaze.hrd.scenario import make_scenario_id
from socialmaze.hrd.solver import analyze

from .brute_force import brute_force_solve


def test_generate_dataset_uniform_full():
    cfg = GameConfig.create(6, "full")
    scenarios, stats = generate_dataset(cfg, 12, seed=0)
    assert len(scenarios) == 12
    assert [s.id for s in scenarios] == [make_scenario_id(cfg, i) for i in range(1, 13)]
    assert Counter(s.player1_role for s in scenarios) == {r: 3 for r in cfg.roles_present()}
    attempts = [s.meta["attempt"] for s in scenarios]
    assert attempts == sorted(attempts) and len(set(attempts)) == 12
    for s in scenarios:
        s.validate()
        assert s.has_complete_roles()
        assert s.solution["unique"]
        assert (s.solution["criminal"], s.solution["player1_role"]) == (s.criminal, s.player1_role)
        assert s.solution == analyze(s)
        assert brute_force_solve(s.config, s.displayed_role, s.all_statements())["unique"]
        assert s.reasoning.endswith(answer_block(s.criminal, s.player1_role))
        assert s.meta["generator_version"] == __version__
        assert s.meta["targeting"] == "random" and s.meta["seed"] == 0
    assert stats["accepted"] == 12 and stats["attempts"] >= 12
    assert stats["acceptance_rate"] == pytest.approx(12 / stats["attempts"])
    assert set(stats["per_role"]) == set(cfg.roles_present())
    assert sum(c["attempts"] for c in stats["per_role"].values()) == stats["attempts"]
    assert all(c["accepted"] == 3 for c in stats["per_role"].values())
    assert stats["p1_role_counts"] == {r: 3 for r in cfg.roles_present()}
    assert sum(stats["solvable_after_round"].values()) == 12
    assert set(stats["solvable_after_round"]) <= {"1", "2", "3"}
    assert 0.0 <= stats["cross_checked_rate"] <= 1.0
    again, again_stats = generate_dataset(cfg, 12, seed=0)
    assert [s.to_dict() for s in again] == [s.to_dict() for s in scenarios]
    assert again_stats == stats
    other, _ = generate_dataset(cfg, 12, seed=1)
    assert [s.to_dict() for s in other] != [s.to_dict() for s in scenarios]


def test_parse_role_mix():
    cfg = GameConfig.create(6, "rumormonger")
    assert parse_role_mix("uniform", cfg) == {INVESTIGATOR: 1 / 3, CRIMINAL: 1 / 3, RUMORMONGER: 1 / 3}
    assert parse_role_mix("natural", cfg) is None
    assert parse_role_mix("Investigator=1,Criminal=1,Rumormonger=2,Lunatic=0", cfg) == {
        INVESTIGATOR: 0.25, CRIMINAL: 0.25, RUMORMONGER: 0.5,
    }
    assert parse_role_mix(" investigator = 3 , criminal = 1 ", cfg) == {INVESTIGATOR: 0.75, CRIMINAL: 0.25}
    with pytest.raises(ValueError):
        parse_role_mix("Investigator=1,Lunatic=1", cfg)
    with pytest.raises(ValueError):
        parse_role_mix("Investigator=0,Criminal=0", cfg)
    with pytest.raises(ValueError):
        parse_role_mix("Detective=1", cfg)
    with pytest.raises(ValueError):
        parse_role_mix("Investigator", cfg)
    with pytest.raises(ValueError):
        parse_role_mix("Investigator=-1,Criminal=2", cfg)
    with pytest.raises(ValueError):
        parse_role_mix("Investigator=lots", cfg)


def test_allocate_quotas_uses_largest_remainder():
    third = {INVESTIGATOR: 1 / 3, CRIMINAL: 1 / 3, RUMORMONGER: 1 / 3}
    assert allocate_quotas(third, 50) == {INVESTIGATOR: 17, CRIMINAL: 17, RUMORMONGER: 16}
    assert allocate_quotas(third, 12) == {INVESTIGATOR: 4, CRIMINAL: 4, RUMORMONGER: 4}
    assert allocate_quotas({CRIMINAL: 0.5, LUNATIC: 0.5}, 7) == {CRIMINAL: 4, LUNATIC: 3}
    assert allocate_quotas({INVESTIGATOR: 0.9, LUNATIC: 0.1}, 3) == {INVESTIGATOR: 3, LUNATIC: 0}


def test_natural_and_explicit_role_mixes():
    cfg = GameConfig.create(6, "full")
    natural, stats = generate_dataset(cfg, 10, role_mix="natural", seed=1)
    assert len(natural) == 10
    assert sum(stats["p1_role_counts"].values()) == 10
    explicit, stats = generate_dataset(cfg, 10, role_mix="Criminal=1,Lunatic=1", seed=1)
    assert Counter(s.player1_role for s in explicit) == {CRIMINAL: 5, LUNATIC: 5}
    assert stats["per_role"][INVESTIGATOR] == {"attempts": 0, "accepted": 0, "acceptance_rate": None}
    with pytest.raises(ValueError):
        generate_dataset(GameConfig.create(6, "original"), 4, role_mix="Lunatic=1")
    with pytest.raises(ValueError):
        generate_dataset(cfg, -1)


def test_strategic_targeting_and_id_prefix():
    scenarios, stats = generate_dataset(GameConfig.create(6, "full"), 4, targeting="strategic", seed=2, id_prefix="demo")
    assert all(s.meta["targeting"] == "strategic" for s in scenarios)
    assert [s.id for s in scenarios] == [f"demo-n6-full-{i:05d}" for i in range(1, 5)]
    assert stats["accepted"] == 4


def test_attempt_limit_raises():
    with pytest.raises(RuntimeError):
        generate_dataset(GameConfig.create(10, "full"), 5, seed=0, max_attempts_factor=0)
    assert generate_dataset(GameConfig.create(6, "full"), 0, seed=0)[0] == []


def test_cli_round_trip(tmp_path, capsys):
    out = tmp_path / "hrd_n6_full.jsonl"
    base = ["generate", "-n", "6", "--variant", "full", "-N", "6", "--seed", "0", "--out", str(out), "--quiet"]
    assert main(base) == 0
    assert "wrote 6 scenarios" in capsys.readouterr().out
    assert out.exists() and meta_path(out).exists()
    meta = load_meta(out)
    assert meta["task"] == "Hidden Role Deduction"
    assert meta["generator_version"] == __version__
    assert meta["config"] == GameConfig.create(6, "full").to_dict()
    assert meta["num_scenarios"] == 6 and meta["seed"] == 0
    assert meta["targeting"] == "random" and meta["role_mix"] == "uniform"
    assert meta["stats"]["accepted"] == 6
    assert meta["command"].startswith("python -m socialmaze.hrd generate -n 6 --variant full")
    assert "--role-mix uniform" in meta["command"]

    # Refuses to overwrite unless asked.
    assert main(base) == 1
    assert "exists" in capsys.readouterr().out
    assert main(base + ["--overwrite"]) == 0
    capsys.readouterr()

    # Bad configurations are reported, not raised.
    assert main(["generate", "-n", "6", "--variant", "original", "--role-mix", "Lunatic=1",
                 "--out", str(tmp_path / "x.jsonl"), "--quiet"]) == 1
    assert "error:" in capsys.readouterr().out

    scenarios = load_scenarios(out)
    assert [s.id for s in scenarios] == [f"hrd-n6-full-{i:05d}" for i in range(1, 7)]
    for s in scenarios:
        s.validate()
        assert s.solution == analyze(s)

    assert main(["solve", str(out)]) == 0
    text = capsys.readouterr().out
    assert "uniquely solvable: 6/6 (100.0%)" in text
    assert "stored answers consistent with the solver: 6/6" in text
    assert "stored solutions equal to the recomputed ones: 6/6" in text
    assert main(["solve", str(out), "--json", "--limit", "4"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["scenarios"] == 4 and stats["unique"] == 4
    assert stats["answer_disagreements"] == [] and stats["solution_disagreements"] == []
    assert main(["solve", str(out), "--explain", "hrd-n6-full-00002"]) == 0
    explained = capsys.readouterr().out.rstrip("\n")
    assert explained == scenarios[1].reasoning
    assert main(["solve", str(out), "--index", "0"]) == 0
    assert capsys.readouterr().out.rstrip("\n") == scenarios[0].reasoning
    assert main(["solve", str(out), "--explain", "nope"]) == 1
    capsys.readouterr()
    assert main(["inspect", str(out), "--prompt"]) == 0
    assert "system prompt" in capsys.readouterr().out
    assert main(["inspect", str(out), "--index", "2", "--reasoning", "--prompt", "--mode", "final"]) == 0
    assert scenarios[2].reasoning in capsys.readouterr().out

    hf = tmp_path / "rows.jsonl"
    assert main(["export", str(out), "--format", "hf", "--out", str(hf)]) == 0
    capsys.readouterr()
    back = load_scenarios(hf)
    assert len(back) == 6
    for a, b in zip(back, scenarios):
        assert a.id == b.id and a.config == b.config and a.displayed_role == b.displayed_role
        assert a.rounds == b.rounds and a.answer == b.answer and a.reasoning == b.reasoning
        assert a.roles == {1: b.player1_role, b.criminal: CRIMINAL}
        row = to_hf_row(b)
        assert row["answer"] == answer_block(b.criminal, b.player1_role, header=False)
        assert from_hf_row(row).rounds == b.rounds


def test_solve_command_flags_wrong_answers(tmp_path, capsys):
    scenarios, _ = generate_dataset(GameConfig.create(6, "full"), 3, seed=3)
    records = [s.to_dict() for s in scenarios]
    wrong = next(p for p in range(2, 7) if p != records[0]["answer"]["criminal"])
    records[0]["answer"]["criminal"] = wrong
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, records)
    assert main(["solve", str(path)]) == 1
    text = capsys.readouterr().out
    assert "stored answers consistent with the solver: 2/3" in text
    assert records[0]["id"] in text
