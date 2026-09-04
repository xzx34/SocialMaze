"""Accuracy, confidence intervals and error decomposition for evaluation results.

The unit of analysis is a result record (one scenario evaluated with one
seed, schema in :mod:`socialmaze.hrd.evaluate`) and, within it, the round
entries. Three accuracies are reported everywhere:

* ``crim``: the predicted Criminal equals the true Criminal;
* ``self``: the predicted role equals Player 1's true role (strict: "Unknown",
  a hedged line and a missing line are all wrong);
* ``both``: both of the above.

Each proportion comes with the 95% normal-approximation binomial half-width
``1.96 * sqrt(p (1 - p) / n)`` used in the paper (about 4.4 points at n = 500
and p = 0.5).

Rounds. In incremental mode a record has one entry per round; in final mode
a single entry at the last round. If a call failed, the evaluator stops the
scenario, so later entries are missing: :func:`entry_for_round` carries such
a failure forward, which keeps the population of the final round equal to
the number of records. Rounds that were never queried (final mode) are not
counted.

Error decomposition. Every record is put into exactly one category at its
final round: ``correct`` (both correct), else the first that applies of
``api_error`` (the call failed), ``truncated`` (the reply hit the completion
cap), ``extraction_failed`` (no criminal line), ``hedged`` (two roles on the
role line), ``reasoning_error`` (a well-formed but wrong answer). Because
``correct`` is checked first, its share equals ``both`` at the final round and
the fractions sum to one.

Everything returned by :func:`summarize_records` is a plain dict of plain
numbers and strings, so it can be written to ``summary.json`` unchanged.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Optional, Sequence

from .rules import UNKNOWN

Z_95 = 1.96

#: Mutually exclusive outcome categories, in priority order after ``correct``.
ERROR_CATEGORIES: tuple[str, ...] = (
    "api_error", "truncated", "extraction_failed", "hedged", "reasoning_error", "correct",
)

#: ``(metric prefix, round-entry flag)`` for the three accuracies.
ACCURACIES: tuple[tuple[str, str], ...] = (
    ("crim", "criminal_correct"),
    ("self", "role_correct"),
    ("both", "both_correct"),
)


# --------------------------------------------------------------------------
# Proportions
# --------------------------------------------------------------------------


def binomial_halfwidth(p: float, n: int) -> float:
    """Half-width of the 95% normal-approximation interval; 0 when ``n == 0``."""
    if n <= 0:
        return 0.0
    return Z_95 * math.sqrt(p * (1.0 - p) / n)


def proportion(count: int, n: int) -> tuple[float, float]:
    """``(p, half-width)`` for ``count`` successes out of ``n``."""
    if n <= 0:
        return 0.0, 0.0
    p = count / n
    return p, binomial_halfwidth(p, n)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------
# Round entries
# --------------------------------------------------------------------------


def record_final_round(record: dict) -> int:
    """The last round of the game (``num_rounds``, else the last entry's round)."""
    if record.get("num_rounds"):
        return int(record["num_rounds"])
    return max(int(e["round"]) for e in record["rounds"])


def failed_entry(round_index: int, error: str) -> dict:
    """A synthetic entry standing for a round never reached because of ``error``."""
    return {
        "round": round_index, "response": "", "reasoning_text": None,
        "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
        "latency_s": None, "pred_criminal": None, "pred_role": None, "found": False,
        "hedged": False, "criminal_correct": False, "role_correct": False,
        "both_correct": False, "truncated": False, "extraction_failed": False,
        "error": error,
    }


def entry_for_round(record: dict, round_index: int) -> Optional[dict]:
    """The record's entry for ``round_index``, carrying an earlier failure forward.

    Returns the stored entry if present; a :func:`failed_entry` if an earlier
    round failed and ``round_index`` is within the game; ``None`` if the round
    was never queried (final mode, or beyond the game's length).
    """
    for entry in record["rounds"]:
        if int(entry["round"]) == round_index:
            return entry
    if round_index > record_final_round(record):
        return None
    for entry in record["rounds"]:
        if entry.get("error") and int(entry["round"]) < round_index:
            return failed_entry(round_index, entry["error"])
    return None


def final_entry(record: dict) -> Optional[dict]:
    """The entry at the record's final round (failures carried forward)."""
    return entry_for_round(record, record_final_round(record))


def classify_entry(entry: dict) -> str:
    """The outcome category of one round entry; see the module docstring."""
    if entry.get("error"):
        return "api_error"
    if entry.get("both_correct"):
        return "correct"
    if entry.get("truncated"):
        return "truncated"
    if not entry.get("found"):
        return "extraction_failed"
    if entry.get("hedged"):
        return "hedged"
    return "reasoning_error"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def accuracy_stats(entries: Sequence[dict]) -> dict:
    """``n``, the three accuracies with half-widths, and the Unknown rate."""
    n = len(entries)
    out: dict = {"n": n}
    for prefix, flag in ACCURACIES:
        p, hw = proportion(sum(1 for e in entries if e.get(flag)), n)
        out[f"{prefix}_acc"] = p
        out[f"{prefix}_halfwidth"] = hw
    out["unknown_rate"] = proportion(sum(1 for e in entries if e.get("pred_role") == UNKNOWN), n)[0]
    return out


def error_decomposition(entries: Sequence[dict]) -> dict:
    """Counts and fractions per category; fractions sum to one when ``n > 0``."""
    counts = {category: 0 for category in ERROR_CATEGORIES}
    for entry in entries:
        counts[classify_entry(entry)] += 1
    n = len(entries)
    fractions = {c: (counts[c] / n if n else 0.0) for c in ERROR_CATEGORIES}
    return {"n": n, "counts": counts, "fractions": fractions}


def usage_stats(final_entries: Sequence[dict], all_entries: Sequence[dict]) -> dict:
    """Mean tokens and latency at the final round plus totals over all rounds."""
    def values(entries: Iterable[dict], key: str) -> list[float]:
        return [float(e[key]) for e in entries if e.get(key) is not None]

    prompt = values(final_entries, "prompt_tokens")
    completion = values(final_entries, "completion_tokens")
    latency = values(final_entries, "latency_s")
    return {
        "n": len(final_entries),
        "prompt_tokens_mean": _mean(prompt),
        "completion_tokens_mean": _mean(completion),
        "latency_s_mean": _mean(latency),
        "prompt_tokens_total": int(sum(values(all_entries, "prompt_tokens"))),
        "completion_tokens_total": int(sum(values(all_entries, "completion_tokens"))),
    }


def seed_variation(records: Sequence[dict]) -> tuple[dict, dict, Optional[dict]]:
    """Per-seed final-round accuracies, their mean, and the sample std (``None`` for one seed)."""
    by_seed: dict[str, dict] = {}
    for seed in sorted({int(r["seed"]) for r in records}):
        entries = [e for r in records if int(r["seed"]) == seed for e in [final_entry(r)] if e is not None]
        by_seed[str(seed)] = accuracy_stats(entries)
    keys = [f"{prefix}_acc" for prefix, _ in ACCURACIES]
    columns = {k: [s[k] for s in by_seed.values()] for k in keys}
    mean = {k: _mean(v) for k, v in columns.items()}
    std = {k: statistics.stdev(v) for k, v in columns.items()} if len(by_seed) > 1 else None
    return by_seed, mean, std


def summarize_records(records: Iterable[dict]) -> dict:
    """Aggregate result records of one model into a JSON-serialisable summary.

    Keys: ``model``, ``mode``, ``num_records``, ``num_scenarios``, ``seeds``,
    ``num_seeds``, ``final_round``, ``rounds`` (round index as string ->
    :func:`accuracy_stats`), ``by_role`` (Player 1 true role -> stats at the
    final round), ``errors`` (:func:`error_decomposition` at the final round),
    ``by_seed``, ``seed_mean``, ``seed_std`` and ``usage``.
    """
    records = list(records)
    finals = [(r, final_entry(r)) for r in records]
    final_entries = [e for _, e in finals if e is not None]
    max_round = max((record_final_round(r) for r in records), default=0)

    rounds: dict[str, dict] = {}
    for t in range(1, max_round + 1):
        entries = [e for r in records for e in [entry_for_round(r, t)] if e is not None]
        if entries:
            rounds[str(t)] = accuracy_stats(entries)

    by_role: dict[str, dict] = {}
    for role in sorted({r["player1_role"] for r in records}):
        by_role[role] = accuracy_stats([e for r, e in finals if e is not None and r["player1_role"] == role])

    by_seed, seed_mean, seed_std = seed_variation(records) if records else ({}, {}, None)
    return {
        "model": records[0].get("model") if records else None,
        "mode": records[0].get("mode") if records else None,
        "num_records": len(records),
        "num_scenarios": len({r["id"] for r in records}),
        "seeds": sorted({int(r["seed"]) for r in records}),
        "num_seeds": len({int(r["seed"]) for r in records}),
        "final_round": max_round or None,
        "rounds": rounds,
        "by_role": by_role,
        "errors": error_decomposition(final_entries),
        "by_seed": by_seed,
        "seed_mean": seed_mean,
        "seed_std": seed_std,
        "usage": usage_stats(final_entries, [e for r in records for e in r["rounds"]]),
    }


__all__ = [
    "Z_95", "ERROR_CATEGORIES", "ACCURACIES", "binomial_halfwidth", "proportion",
    "record_final_round", "failed_entry", "entry_for_round", "final_entry",
    "classify_entry", "accuracy_stats", "error_decomposition", "usage_stats",
    "seed_variation", "summarize_records",
]
