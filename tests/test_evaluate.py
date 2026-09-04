"""The evaluator with the offline mock client: scoring, modes, resume, retries, seeds, CLI."""

import json
from pathlib import Path

import pytest

from socialmaze.hrd import cli, prompts
from socialmaze.hrd.evaluate import (
    FINAL,
    INCREMENTAL,
    evaluate,
    evaluate_scenario,
    has_error,
    safe_name,
    score_round,
)
from socialmaze.hrd.io import load_scenarios, read_jsonl
from socialmaze.hrd.metrics import classify_entry, summarize_records
from socialmaze.hrd.rules import UNKNOWN
from socialmaze.llm.client import ChatResult, ModelSpec
from socialmaze.llm.mock import TRANSIENT_ERROR, MockClient

LEGACY = Path(__file__).resolve().parents[1] / "archive" / "hidden_role_deduction" / "data" / "hrd_6_all.json"


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios(LEGACY)


def spec(name="mock", behaviour="oracle"):
    return ModelSpec(name=name, provider="mock", model=behaviour)


def run(scenarios, behaviour="oracle", client=None, **kwargs):
    client = client if client is not None else MockClient(behaviour)
    kwargs.setdefault("progress", False)
    kwargs.setdefault("workers", 4)
    records = evaluate(scenarios, spec(f"mock:{behaviour}", behaviour), client=client, **kwargs)
    return records, client


class RecordingClient(MockClient):
    """An oracle that keeps a copy of every conversation it was sent."""

    def __init__(self):
        super().__init__("oracle")
        self.conversations = []

    def chat(self, messages, temperature, max_tokens=None, context=None):
        self.conversations.append([dict(m) for m in messages])
        return super().chat(messages, temperature, max_tokens, context)


# -- scoring and modes -----------------------------------------------------


def test_oracle_incremental_is_perfect_every_round(scenarios):
    records, client = run(scenarios)
    assert len(records) == len(scenarios) == 10
    assert client.calls == 30
    for rec, sc in zip(records, scenarios):
        assert rec["id"] == sc.id and rec["seed"] == 0 and rec["model"] == "mock:oracle"
        assert rec["mode"] == INCREMENTAL and rec["num_rounds"] == 3 and rec["num_players"] == 6
        assert rec["answer"] == sc.answer.to_dict() and rec["player1_role"] == sc.player1_role
        assert [e["round"] for e in rec["rounds"]] == [1, 2, 3]
        for e in rec["rounds"]:
            assert e["both_correct"] and e["criminal_correct"] and e["role_correct"]
            assert e["pred_criminal"] == sc.criminal and e["pred_role"] == sc.player1_role
            assert e["error"] is None and not e["truncated"] and not e["extraction_failed"]
            assert e["prompt_tokens"] > 0 and e["completion_tokens"] > 0
    summary = summarize_records(records)
    assert list(summary["rounds"]) == ["1", "2", "3"]
    assert all(summary["rounds"][t]["both_acc"] == 1.0 for t in ("1", "2", "3"))
    assert summary["errors"]["fractions"]["correct"] == 1.0


def test_incremental_conversation_shape(scenarios):
    sc = scenarios[0]
    client = RecordingClient()
    evaluate_scenario(client, sc, INCREMENTAL, 0.7, None, 0, "mock")
    assert len(client.conversations) == 3
    first, second, third = client.conversations
    assert first[0] == {"role": "system", "content": prompts.system_prompt(sc.config, sc.displayed_role)}
    assert first[1] == {"role": "user", "content": prompts.round_message(1, sc.rounds[0])}
    assert [m["role"] for m in third] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert third[3]["content"] == prompts.round_message(2, sc.rounds[1])
    assert third[5]["content"] == prompts.round_message(3, sc.rounds[2])
    assert "Final Criminal Is Player" in third[2]["content"]


def test_final_mode_single_round(scenarios):
    sc = scenarios[0]
    client = RecordingClient()
    rec = evaluate_scenario(client, sc, FINAL, 0.7, None, 0, "mock")
    assert len(client.conversations) == 1
    assert client.conversations[0][1] == {"role": "user", "content": prompts.final_message(sc.rounds)}
    assert [e["round"] for e in rec["rounds"]] == [3]
    assert rec["rounds"][0]["both_correct"]
    records, client = run(scenarios, mode=FINAL)
    assert client.calls == 10
    summary = summarize_records(records)
    assert list(summary["rounds"]) == ["3"] and summary["rounds"]["3"]["both_acc"] == 1.0


def test_wrong_mock_scores_zero(scenarios):
    records, _ = run(scenarios, "wrong")
    summary = summarize_records(records)
    for t in ("1", "2", "3"):
        stats = summary["rounds"][t]
        assert stats["crim_acc"] == 0.0 and stats["self_acc"] == 0.0 and stats["both_acc"] == 0.0
    assert summary["errors"]["fractions"]["reasoning_error"] == 1.0
    for rec in records:
        assert all(e["found"] and 1 <= e["pred_criminal"] <= 6 for e in rec["rounds"])


def test_unknown_mock(scenarios):
    records, _ = run(scenarios, "unknown")
    summary = summarize_records(records)
    stats = summary["rounds"]["3"]
    assert stats["crim_acc"] == 1.0 and stats["self_acc"] == 0.0 and stats["both_acc"] == 0.0
    assert stats["unknown_rate"] == 1.0
    assert all(e["pred_role"] == UNKNOWN for rec in records for e in rec["rounds"])
    assert summary["errors"]["fractions"]["reasoning_error"] == 1.0


def test_garbage_mock_is_extraction_failure(scenarios):
    records, _ = run(scenarios, "garbage")
    for rec in records:
        for e in rec["rounds"]:
            assert e["extraction_failed"] and not e["found"] and e["pred_criminal"] is None
            assert e["error"] is None and not e["truncated"]
            assert classify_entry(e) == "extraction_failed"
    assert summarize_records(records)["errors"]["fractions"]["extraction_failed"] == 1.0


def test_truncate_mock(scenarios):
    records, _ = run(scenarios, "truncate")
    for rec in records:
        for e in rec["rounds"]:
            assert e["truncated"] and e["finish_reason"] == "length" and not e["found"]
            assert classify_entry(e) == "truncated"
    assert summarize_records(records)["errors"]["fractions"]["truncated"] == 1.0


def test_hedged_mock(scenarios):
    records, _ = run(scenarios, "hedged")
    for rec in records:
        for e in rec["rounds"]:
            assert e["hedged"] and e["found"] and e["criminal_correct"]
            assert e["pred_role"] is None and not e["role_correct"] and not e["both_correct"]
            assert classify_entry(e) == "hedged"
    summary = summarize_records(records)
    assert summary["errors"]["fractions"]["hedged"] == 1.0
    assert summary["rounds"]["3"]["crim_acc"] == 1.0 and summary["rounds"]["3"]["self_acc"] == 0.0


def test_score_round_with_error_and_invalid_mode(scenarios):
    sc = scenarios[0]
    entry = score_round(2, ChatResult(text="", error="MockError: boom"), sc.answer)
    assert entry["round"] == 2 and entry["error"] == "MockError: boom"
    assert not entry["found"] and not entry["truncated"] and not entry["extraction_failed"]
    assert not entry["both_correct"] and entry["pred_role"] is None
    with pytest.raises(ValueError):
        evaluate_scenario(MockClient(), sc, "sometimes", 0.7, None, 0)
    with pytest.raises(ValueError):
        evaluate([sc], spec(), mode="sometimes", client=MockClient(), progress=False)


# -- resume, retries, seeds ------------------------------------------------


def test_resume_skips_finished_pairs(scenarios, tmp_path):
    out = tmp_path / "mock.jsonl"
    first, client1 = run(scenarios, out_path=out)
    assert client1.calls == 30 and len(read_jsonl(out)) == 10
    second, client2 = run(scenarios, out_path=out)
    assert client2.calls == 0
    assert len(second) == 10 and [r["id"] for r in second] == [s.id for s in scenarios]
    pairs = [(r["id"], r["seed"]) for r in read_jsonl(out)]
    assert len(pairs) == len(set(pairs)) == 10
    # Adding a scenario later queries only that one.
    more = load_scenarios(LEGACY.with_name("hrd_6_lunatic.json"), limit=2)
    third, client3 = run(scenarios + more, out_path=out)
    assert client3.calls == 6 and len(third) == 12 and len(read_jsonl(out)) == 12


def test_no_resume_starts_over(scenarios, tmp_path):
    out = tmp_path / "mock.jsonl"
    run(scenarios, out_path=out)
    records, client = run(scenarios, out_path=out, resume=False)
    assert client.calls == 30 and len(read_jsonl(out)) == 10 and len(records) == 10


def test_resume_refuses_foreign_records(scenarios, tmp_path):
    out = tmp_path / "mock.jsonl"
    records, _ = run(scenarios, out_path=out)
    foreign = dict(records[0], model="gpt-4o")
    out.write_text(json.dumps(foreign) + "\n")
    with pytest.raises(ValueError, match="gpt-4o"):
        run(scenarios, out_path=out)


def test_retry_errors_with_flaky_mock_final_mode(scenarios, tmp_path):
    out = tmp_path / "flaky.jsonl"
    client = MockClient("flaky")
    first, _ = run(scenarios, client=client, out_path=out, mode=FINAL)
    assert all(has_error(r) for r in first)
    assert all(r["rounds"][0]["error"] == TRANSIENT_ERROR for r in first)
    assert summarize_records(first)["errors"]["fractions"]["api_error"] == 1.0
    second, _ = run(scenarios, client=client, out_path=out, mode=FINAL)
    assert client.calls == 20
    assert not any(has_error(r) for r in second)
    assert all(r["rounds"][0]["both_correct"] for r in second)
    stored = read_jsonl(out)
    assert len(stored) == 10 and not any(has_error(r) for r in stored)
    pairs = [(r["id"], r["seed"]) for r in stored]
    assert len(set(pairs)) == 10


def test_retry_errors_with_flaky_mock_incremental_converges(scenarios, tmp_path):
    out = tmp_path / "flaky.jsonl"
    client = MockClient("flaky")
    records, _ = run(scenarios, client=client, out_path=out)
    assert all(has_error(r) for r in records)
    assert all(len(r["rounds"]) == 1 for r in records), "a failed round stops the scenario"
    runs = 1
    while any(has_error(r) for r in records):
        records, _ = run(scenarios, client=client, out_path=out)
        runs += 1
        assert runs <= 4
    assert runs == 4, "one round of the three fails per run, then a clean run"
    assert all(len(r["rounds"]) == 3 and r["rounds"][2]["both_correct"] for r in records)
    pairs = [(r["id"], r["seed"]) for r in read_jsonl(out)]
    assert len(pairs) == len(set(pairs)) == 10


def test_keep_errors_when_retry_is_off(scenarios, tmp_path):
    out = tmp_path / "flaky.jsonl"
    client = MockClient("flaky")
    run(scenarios, client=client, out_path=out, mode=FINAL)
    records, _ = run(scenarios, client=client, out_path=out, mode=FINAL, retry_errors=False)
    assert client.calls == 10 and all(has_error(r) for r in records)


def test_seeds_give_one_record_per_scenario_and_seed(scenarios, tmp_path):
    out = tmp_path / "mock.jsonl"
    records, client = run(scenarios, out_path=out, seeds=2)
    assert client.calls == 60 and len(records) == 20
    assert [(r["id"], r["seed"]) for r in records][:4] == [
        (scenarios[0].id, 0), (scenarios[0].id, 1), (scenarios[1].id, 0), (scenarios[1].id, 1)]
    summary = summarize_records(records)
    assert summary["num_seeds"] == 2 and summary["num_scenarios"] == 10
    assert summary["rounds"]["3"]["n"] == 20 and summary["seed_std"]["both_acc"] == 0.0
    again, client2 = run(scenarios, out_path=out, seeds=3)
    assert client2.calls == 30 and len(again) == 30


def test_evaluate_without_output_path(scenarios):
    records, _ = run(scenarios[:3], workers=1)
    assert len(records) == 3


# -- command line ----------------------------------------------------------


def test_safe_name():
    assert safe_name("gpt-4o") == "gpt-4o"
    assert safe_name("openrouter/anthropic/claude-sonnet-4.5") == "openrouter__anthropic__claude-sonnet-4.5"
    assert safe_name("mock:wrong") == "mock__wrong"


def test_cli_evaluate_writes_run_directory(tmp_path, capsys):
    out = tmp_path / "run"
    code = cli.main(["evaluate", "--data", str(LEGACY), "--models", "mock", "mock-wrong",
                     "--limit", "5", "--out", str(out), "--quiet"])
    assert code == 0
    assert (out / "run.json").exists() and (out / "summary.json").exists() and (out / "report.md").exists()
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["mock-wrong.jsonl", "mock.jsonl"]
    assert len(read_jsonl(out / "mock.jsonl")) == 5 and len(read_jsonl(out / "mock-wrong.jsonl")) == 5
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["models"] == ["mock", "mock-wrong"] and manifest["limit"] == 5
    assert manifest["mode"] == "incremental" and manifest["temperature"] == 0.7 and manifest["seeds"] == 1
    assert len(manifest["scenario_ids"]) == 5 and manifest["socialmaze_version"]
    assert manifest["model_specs"]["mock"]["provider"] == "mock"
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["models"]) == {"mock", "mock-wrong"}
    assert summary["models"]["mock"]["rounds"]["3"]["both_acc"] == 1.0
    assert summary["models"]["mock-wrong"]["rounds"]["3"]["both_acc"] == 0.0
    captured = capsys.readouterr().out
    assert "mock: round 3 (n=5): both 100.0" in captured
    assert "mock-wrong: round 3 (n=5): both 0.0" in captured
    assert "# SocialMaze" not in captured, "--quiet suppresses the report"

    # Running again resumes: the third model is added, the first two are not re-queried.
    code = cli.main(["evaluate", "--data", str(LEGACY), "--models", "mock", "mock:unknown",
                     "--limit", "5", "--out", str(out), "--mode", "incremental"])
    assert code == 0
    assert json.loads((out / "run.json").read_text())["models"] == ["mock", "mock-wrong", "mock:unknown"]
    assert len(read_jsonl(out / "mock.jsonl")) == 5
    assert (out / "mock__unknown.jsonl").exists()
    report = (out / "report.md").read_text()
    assert "mock:unknown" in report and "mock-wrong" in report
    assert "# SocialMaze" in capsys.readouterr().out


def test_cli_evaluate_unknown_model_fails_before_running(tmp_path):
    with pytest.raises(ValueError):
        cli.main(["evaluate", "--data", str(LEGACY), "--models", "no-such-model",
                  "--out", str(tmp_path / "run"), "--quiet"])
    assert not (tmp_path / "run").exists()
