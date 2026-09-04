"""Accuracy, confidence intervals and the error decomposition on hand-built records."""

import json
import statistics

import pytest

from socialmaze.hrd.metrics import (
    ERROR_CATEGORIES,
    accuracy_stats,
    binomial_halfwidth,
    classify_entry,
    entry_for_round,
    error_decomposition,
    final_entry,
    proportion,
    summarize_records,
)
from socialmaze.hrd.rules import LUNATIC, RUMORMONGER, UNKNOWN


def entry(t, crim=True, role=True, found=True, hedged=False, truncated=False, error=None,
          pred_role=None, tokens=(100, 50), latency=1.0):
    """A round entry as :func:`socialmaze.hrd.evaluate.score_round` produces it."""
    role_ok = bool(role and found and not hedged)
    crim_ok = bool(crim and found)
    return {
        "round": t, "response": "", "reasoning_text": None,
        "finish_reason": "length" if truncated else "stop",
        "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "latency_s": latency,
        "pred_criminal": 4 if found else None,
        "pred_role": pred_role if pred_role is not None else (LUNATIC if role_ok else None),
        "found": found, "hedged": hedged,
        "criminal_correct": crim_ok, "role_correct": role_ok, "both_correct": crim_ok and role_ok,
        "truncated": truncated, "extraction_failed": not found, "error": error,
    }


def record(sid, entries, seed=0, role=LUNATIC, num_rounds=3):
    return {
        "id": sid, "seed": seed, "model": "m", "mode": "incremental", "num_players": 6,
        "num_rounds": num_rounds, "variant": "full", "displayed_role": "Criminal",
        "player1_role": role, "answer": {"criminal": 4, "player1_role": role},
        "rounds": entries, "created": "2026-01-01T00:00:00+00:00",
    }


def full(sid, seed=0, player_role=LUNATIC, **flags):
    """A complete three-round record; ``flags`` go to every :func:`entry`."""
    return record(sid, [entry(t, **flags) for t in (1, 2, 3)], seed=seed, role=player_role)


def test_binomial_halfwidth():
    assert binomial_halfwidth(0.5, 500) == pytest.approx(0.0438, abs=1e-3)
    assert binomial_halfwidth(0.5, 0) == 0.0
    assert binomial_halfwidth(0.0, 100) == 0.0
    assert binomial_halfwidth(1.0, 100) == 0.0
    assert binomial_halfwidth(0.9, 100) < binomial_halfwidth(0.5, 100)


def test_proportion():
    p, hw = proportion(3, 4)
    assert p == 0.75 and hw == pytest.approx(1.96 * (0.75 * 0.25 / 4) ** 0.5)
    assert proportion(0, 0) == (0.0, 0.0)


def test_classify_entry_priority():
    assert classify_entry(entry(1, error="APIConnectionError: x", found=False)) == "api_error"
    assert classify_entry(entry(1, truncated=True, found=False)) == "truncated"
    assert classify_entry(entry(1, found=False)) == "extraction_failed"
    assert classify_entry(entry(1, hedged=True)) == "hedged"
    assert classify_entry(entry(1, role=False)) == "reasoning_error"
    assert classify_entry(entry(1, crim=False)) == "reasoning_error"
    assert classify_entry(entry(1)) == "correct"
    # A correct answer is correct even if the cap was hit right after it.
    assert classify_entry(entry(1, truncated=True)) == "correct"


def test_error_decomposition_sums_to_one():
    entries = [
        entry(3, error="boom", found=False), entry(3, truncated=True, found=False),
        entry(3, found=False), entry(3, hedged=True), entry(3, role=False), entry(3), entry(3),
    ]
    dec = error_decomposition(entries)
    assert dec["n"] == 7
    assert sum(dec["fractions"].values()) == pytest.approx(1.0)
    assert dec["counts"] == {"api_error": 1, "truncated": 1, "extraction_failed": 1,
                             "hedged": 1, "reasoning_error": 1, "correct": 2}
    assert tuple(dec["counts"]) == ERROR_CATEGORIES


def test_accuracy_stats_and_unknown_rate():
    stats = accuracy_stats([entry(3), entry(3, role=False, pred_role=UNKNOWN), entry(3, crim=False), entry(3, found=False)])
    assert stats["n"] == 4
    assert stats["crim_acc"] == 0.5 and stats["self_acc"] == 0.5 and stats["both_acc"] == 0.25
    assert stats["both_halfwidth"] == pytest.approx(binomial_halfwidth(0.25, 4))
    assert stats["unknown_rate"] == 0.25
    assert accuracy_stats([]) == {"n": 0, "crim_acc": 0.0, "crim_halfwidth": 0.0, "self_acc": 0.0,
                                  "self_halfwidth": 0.0, "both_acc": 0.0, "both_halfwidth": 0.0,
                                  "unknown_rate": 0.0}


def test_failed_round_is_carried_forward():
    rec = record("a", [entry(1), entry(2, error="RateLimitError: slow", found=False)])
    assert entry_for_round(rec, 1)["both_correct"] is True
    assert entry_for_round(rec, 2)["error"] == "RateLimitError: slow"
    carried = entry_for_round(rec, 3)
    assert carried["error"] == "RateLimitError: slow" and carried["both_correct"] is False
    assert entry_for_round(rec, 4) is None
    assert final_entry(rec)["round"] == 3
    summary = summarize_records([rec, full("b")])
    assert summary["rounds"]["3"]["n"] == 2
    assert summary["rounds"]["3"]["both_acc"] == 0.5
    assert summary["errors"]["counts"]["api_error"] == 1


def test_final_mode_records_only_have_the_last_round():
    rec = record("a", [entry(3)])
    assert entry_for_round(rec, 1) is None and entry_for_round(rec, 2) is None
    failed = record("b", [entry(3, error="boom", found=False)])
    assert entry_for_round(failed, 1) is None
    summary = summarize_records([rec, failed])
    assert list(summary["rounds"]) == ["3"]
    assert summary["rounds"]["3"]["n"] == 2
    assert summary["errors"]["fractions"]["api_error"] == 0.5


def test_per_role_aggregation():
    records = [
        full("a", player_role=LUNATIC),
        full("b", player_role=LUNATIC, role=False),
        full("c", player_role=RUMORMONGER),
        full("d", player_role=RUMORMONGER),
        full("e", player_role=RUMORMONGER, crim=False),
    ]
    summary = summarize_records(records)
    assert set(summary["by_role"]) == {LUNATIC, RUMORMONGER}
    assert summary["by_role"][LUNATIC]["n"] == 2 and summary["by_role"][LUNATIC]["self_acc"] == 0.5
    assert summary["by_role"][RUMORMONGER]["n"] == 3
    assert summary["by_role"][RUMORMONGER]["crim_acc"] == pytest.approx(2 / 3)
    assert summary["by_role"][RUMORMONGER]["both_acc"] == pytest.approx(2 / 3)
    assert summary["num_scenarios"] == 5 and summary["num_seeds"] == 1
    assert summary["seed_std"] is None
    assert summary["errors"]["fractions"]["correct"] == summary["rounds"]["3"]["both_acc"]


def test_per_seed_aggregation():
    records = [full("a", seed=0), full("b", seed=0), full("a", seed=1), full("b", seed=1, role=False)]
    summary = summarize_records(records)
    assert summary["seeds"] == [0, 1] and summary["num_seeds"] == 2 and summary["num_records"] == 4
    assert summary["by_seed"]["0"]["both_acc"] == 1.0
    assert summary["by_seed"]["1"]["both_acc"] == 0.5
    assert summary["seed_mean"]["both_acc"] == 0.75
    assert summary["seed_std"]["both_acc"] == pytest.approx(statistics.stdev([1.0, 0.5]))
    assert summary["seed_std"]["crim_acc"] == 0.0
    assert summary["rounds"]["3"]["n"] == 4


def test_usage_and_round_curve():
    records = [
        record("a", [entry(1, role=False, tokens=(100, 10), latency=0.5), entry(2, tokens=(200, 20), latency=1.0), entry(3, tokens=(300, 30), latency=1.5)]),
        record("b", [entry(1, role=False, tokens=(100, 10)), entry(2, role=False, tokens=(200, 20)), entry(3, tokens=(500, 50), latency=2.5)]),
    ]
    summary = summarize_records(records)
    assert [summary["rounds"][t]["both_acc"] for t in ("1", "2", "3")] == [0.0, 0.5, 1.0]
    usage = summary["usage"]
    assert usage["n"] == 2
    assert usage["prompt_tokens_mean"] == 400 and usage["completion_tokens_mean"] == 40
    assert usage["latency_s_mean"] == 2.0
    assert usage["prompt_tokens_total"] == 1400 and usage["completion_tokens_total"] == 140


def test_summary_is_json_serialisable_and_handles_empty_input():
    summary = summarize_records([full("a"), full("b", seed=1)])
    json.dumps(summary)
    empty = summarize_records([])
    json.dumps(empty)
    assert empty["num_records"] == 0 and empty["rounds"] == {} and empty["final_round"] is None
    assert empty["errors"]["n"] == 0
