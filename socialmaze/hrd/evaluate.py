"""Query models on Hidden Role Deduction scenarios and record the results.

Protocol
--------
Every scenario is played from Player 1's perspective with the system prompt
of :func:`socialmaze.hrd.prompts.system_prompt`. In ``incremental`` mode (the
paper's protocol) the statements of round ``t`` are sent as a user message,
the model answers with reasoning and a Final Judgment, the answer is kept in
the conversation and the next round follows; this gives one scored entry per
round. In ``final`` mode all rounds are sent in one user message and there is
a single entry at the last round. If a call fails (after the client's own
retries) the remaining rounds of that scenario are not queried; the record
keeps the error and is re-queried by the next run unless ``retry_errors`` is
off.

Result record (one per scenario and seed, one JSON object per line in
``<run dir>/<model>.jsonl``)::

    {"id": "hrd-n6-full-00001", "seed": 0, "model": "gpt-4o-mini",
     "mode": "incremental", "num_players": 6, "num_rounds": 3, "variant": "full",
     "displayed_role": "Investigator", "player1_role": "Rumormonger",
     "answer": {"criminal": 4, "player1_role": "Rumormonger"},
     "rounds": [{"round": 1, "response": "...", "reasoning_text": null,
                 "finish_reason": "stop", "prompt_tokens": 812,
                 "completion_tokens": 240, "latency_s": 3.1,
                 "pred_criminal": 4, "pred_role": "Investigator",
                 "found": true, "hedged": false, "criminal_correct": true,
                 "role_correct": false, "both_correct": false,
                 "truncated": false, "extraction_failed": false,
                 "error": null}, ...],
     "created": "2026-09-04T12:00:00+00:00"}

Seeds are independent repetitions ``0 .. seeds-1`` at the same temperature;
the seed value is recorded and passed to the client in ``context`` but is
not sent to the provider, because most providers ignore or reject it.
Scenarios are evaluated by a thread pool; each finished record is appended to
the output file immediately, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

from tqdm import tqdm

from .. import __version__
from ..llm.client import TRUNCATED_FINISH_REASON, BaseClient, ChatResult, ModelSpec, make_client
from ..llm.registry import Registry, load_env
from . import io, prompts, report
from .metrics import summarize_records
from .parsing import parse_final_judgment
from .scenario import Answer, Scenario

INCREMENTAL = "incremental"
FINAL = "final"
MODES: tuple[str, ...] = (INCREMENTAL, FINAL)
RUN_FILE = "run.json"
DEFAULT_TEMPERATURE = 0.7

PathLike = Union[str, Path]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# One scenario
# --------------------------------------------------------------------------


def score_round(round_index: int, result: ChatResult, answer: Answer) -> dict:
    """The round entry for ``result``: the reply, usage, prediction and scores.

    ``role_correct`` is a strict comparison, so "Unknown", a hedged line and
    a missing line are all wrong. ``truncated`` marks replies cut by the
    completion cap and ``extraction_failed`` replies without a criminal
    line; both are only set when the call itself succeeded.
    """
    entry = {
        "round": round_index,
        "response": result.text,
        "reasoning_text": result.reasoning_text,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_s": result.latency_s,
        "pred_criminal": None,
        "pred_role": None,
        "found": False,
        "hedged": False,
        "criminal_correct": False,
        "role_correct": False,
        "both_correct": False,
        "truncated": False,
        "extraction_failed": False,
        "error": result.error,
    }
    if result.error:
        return entry
    pred = parse_final_judgment(result.text)
    entry["pred_criminal"] = pred.criminal
    entry["pred_role"] = pred.role
    entry["found"] = pred.found
    entry["hedged"] = pred.hedged
    entry["criminal_correct"] = pred.criminal == answer.criminal
    entry["role_correct"] = pred.role == answer.player1_role
    entry["both_correct"] = entry["criminal_correct"] and entry["role_correct"]
    entry["truncated"] = result.finish_reason == TRUNCATED_FINISH_REASON
    entry["extraction_failed"] = not pred.found
    return entry


def conversation_turns(scenario: Scenario, mode: str) -> list[tuple[int, str]]:
    """``(round index, user message)`` pairs to send, in order, for ``mode``."""
    if mode == FINAL:
        return [(scenario.num_rounds, prompts.final_message(scenario.rounds))]
    return [(t, prompts.round_message(t, rnd)) for t, rnd in enumerate(scenario.rounds, start=1)]


def evaluate_scenario(
    client: BaseClient,
    scenario: Scenario,
    mode: str = INCREMENTAL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    seed: int = 0,
    model: str = "",
) -> dict:
    """Play one scenario with ``client`` and return its result record."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    messages: list[dict] = [
        {"role": "system", "content": prompts.system_prompt(scenario.config, scenario.displayed_role)}
    ]
    entries: list[dict] = []
    for round_index, user_message in conversation_turns(scenario, mode):
        messages.append({"role": "user", "content": user_message})
        context = {
            "answer": scenario.answer,
            "round": round_index,
            "num_rounds": scenario.num_rounds,
            "num_players": scenario.num_players,
            "scenario_id": scenario.id,
            "seed": seed,
            "mode": mode,
        }
        result = client.chat(messages, temperature, max_tokens, context)
        entries.append(score_round(round_index, result, scenario.answer))
        if not result.ok:
            break
        messages.append({"role": "assistant", "content": result.text})
    return {
        "id": scenario.id,
        "seed": seed,
        "model": model,
        "mode": mode,
        "num_players": scenario.num_players,
        "num_rounds": scenario.num_rounds,
        "variant": scenario.variant,
        "displayed_role": scenario.displayed_role,
        "player1_role": scenario.player1_role,
        "answer": scenario.answer.to_dict(),
        "rounds": entries,
        "created": now_iso(),
    }


# --------------------------------------------------------------------------
# A dataset
# --------------------------------------------------------------------------


def has_error(record: dict) -> bool:
    return any(entry.get("error") for entry in record["rounds"])


def load_existing(out_path: Path, model: str, mode: str, retry_errors: bool) -> list[dict]:
    """Records already in ``out_path`` that can be kept when resuming.

    Records of another model or mode are refused, because mixing them in one
    file would corrupt the report. Records with a failed round are dropped
    (and the file rewritten without them) when ``retry_errors`` is set, so
    that they are queried again. Duplicate ``(id, seed)`` pairs keep the last.
    """
    if not out_path.exists():
        return []
    records = io.read_jsonl(out_path)
    foreign = [r for r in records if r.get("model") != model or r.get("mode") != mode]
    if foreign:
        raise ValueError(
            f"{out_path} holds records of model {foreign[0].get('model')!r} in mode "
            f"{foreign[0].get('mode')!r}; use another output directory or --no-resume"
        )
    kept: dict[tuple, dict] = {}
    for record in records:
        if retry_errors and has_error(record):
            continue
        kept[(record["id"], int(record["seed"]))] = record
    result = list(kept.values())
    if len(result) != len(records):
        io.write_jsonl(out_path, result)
    return result


def sort_records(records: list[dict], scenarios: Sequence[Scenario]) -> list[dict]:
    """Order records by scenario position then seed (unknown ids last)."""
    position = {sc.id: i for i, sc in enumerate(scenarios)}
    return sorted(records, key=lambda r: (position.get(r["id"], len(position)), int(r["seed"])))


def evaluate(
    scenarios: Sequence[Scenario],
    spec: ModelSpec,
    mode: str = INCREMENTAL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    seeds: int = 1,
    workers: int = 8,
    out_path: Optional[PathLike] = None,
    resume: bool = True,
    retry_errors: bool = True,
    progress: bool = True,
    client: Optional[BaseClient] = None,
) -> list[dict]:
    """Evaluate ``spec`` on ``scenarios`` with seeds ``0 .. seeds-1``.

    Returns every record of this model (kept and new). With ``out_path`` each
    finished record is appended under a lock; with ``resume`` the pairs
    already in the file are skipped. ``client`` can be injected (tests);
    otherwise it is built with :func:`make_client`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    client = client if client is not None else make_client(spec)
    out_path = Path(out_path) if out_path is not None else None
    existing: list[dict] = []
    if out_path is not None:
        if resume:
            existing = load_existing(out_path, spec.name, mode, retry_errors)
        else:
            io.write_jsonl(out_path, [])
    done = {(r["id"], int(r["seed"])) for r in existing}
    jobs = [(sc, seed) for sc in scenarios for seed in range(seeds) if (sc.id, seed) not in done]

    lock = threading.Lock()
    new: list[dict] = []

    def run(scenario: Scenario, seed: int) -> dict:
        record = evaluate_scenario(client, scenario, mode, temperature, max_tokens, seed, spec.name)
        with lock:
            if out_path is not None:
                io.append_jsonl(out_path, record)
            new.append(record)
        return record

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(run, sc, seed) for sc, seed in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=spec.name,
                           unit="game", disable=not progress):
            future.result()
    return sort_records(existing + new, scenarios)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def safe_name(name: str) -> str:
    """File stem for a model name: ``openrouter/x/y`` -> ``openrouter__x__y``."""
    return name.replace("/", "__").replace(":", "__")


def default_run_dir() -> Path:
    return Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")


def load_scenarios_for(args: argparse.Namespace) -> list[Scenario]:
    if args.from_hf:
        return io.load_hf(args.from_hf, args.split, args.limit)
    return io.load_scenarios(args.data, args.limit)


def run_manifest(args: argparse.Namespace, specs: Sequence[ModelSpec],
                 scenarios: Sequence[Scenario], previous: Optional[dict]) -> dict:
    """The ``run.json`` contents; models accumulate over resumed invocations."""
    models = list((previous or {}).get("models", []))
    models += [s.name for s in specs if s.name not in models]
    model_specs = dict((previous or {}).get("model_specs", {}))
    model_specs.update({s.name: s.to_dict() for s in specs})
    return {
        "data": str(args.data) if args.data else None,
        "from_hf": args.from_hf or None,
        "split": args.split if args.from_hf else None,
        "limit": args.limit,
        "mode": args.mode,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seeds": args.seeds,
        "models": models,
        "model_specs": model_specs,
        "num_scenarios": len(scenarios),
        "scenario_ids": [sc.id for sc in scenarios],
        "socialmaze_version": __version__,
        "created": (previous or {}).get("created") or now_iso(),
        "updated": now_iso(),
    }


def one_line_summary(name: str, summary: dict) -> str:
    """``model: round 3 (n=500): both 41.2 ± 4.3, self 55.0 ± 4.4, crim 70.2 ± 4.0``."""
    final_round = summary.get("final_round")
    stats = summary["rounds"].get(str(final_round)) if final_round else None
    if not stats:
        return f"{name}: no results"
    parts = ", ".join(
        f"{prefix} {report.format_pct(stats[f'{prefix}_acc'], stats[f'{prefix}_halfwidth'])}"
        for prefix in ("both", "self", "crim")
    )
    return f"{name}: round {final_round} (n={stats['n']}): {parts}"


def run_evaluate(args: argparse.Namespace) -> int:
    """The ``evaluate`` command; see :func:`socialmaze.hrd.cli.build_parser`."""
    load_env()
    scenarios = load_scenarios_for(args)
    for scenario in scenarios:
        scenario.validate()
    registry = Registry.load(args.models_config)
    specs = [registry.resolve(name) for name in args.models]
    clients = [make_client(spec) for spec in specs]

    out_dir = Path(args.out) if args.out else default_run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / RUN_FILE
    previous = None
    if run_path.exists() and not args.no_resume:
        with open(run_path, encoding="utf-8") as f:
            previous = json.load(f)
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest(args, specs, scenarios, previous), f, indent=2, ensure_ascii=False)
        f.write("\n")

    for spec, client in zip(specs, clients):
        records = evaluate(
            scenarios, spec, mode=args.mode, temperature=args.temperature,
            max_tokens=args.max_tokens, seeds=args.seeds, workers=args.workers,
            out_path=out_dir / f"{safe_name(spec.name)}.jsonl",
            resume=not args.no_resume, retry_errors=not args.no_retry_errors,
            progress=not args.quiet, client=client,
        )
        print(one_line_summary(spec.name, summarize_records(records)))

    md_path, _ = report.write_report(out_dir)
    if not args.quiet:
        print(md_path.read_text(encoding="utf-8"))
    return 0


__all__ = [
    "INCREMENTAL", "FINAL", "MODES", "RUN_FILE", "DEFAULT_TEMPERATURE", "now_iso",
    "score_round", "conversation_turns", "evaluate_scenario", "has_error",
    "load_existing", "sort_records", "evaluate", "safe_name", "default_run_dir",
    "load_scenarios_for", "run_manifest", "one_line_summary", "run_evaluate",
]
