"""Command-line interface for Hidden Role Deduction.

Run as ``python -m socialmaze.hrd <command>`` or ``socialmaze-hrd <command>``.

Commands
--------
generate   simulate games, keep the uniquely solvable ones, write a JSONL dataset
solve      re-run the solver on a dataset (uniqueness statistics, ``--explain`` one instance)
inspect    print one scenario (roles, statements, answer, solution) and optionally its prompts
export     convert a dataset to the HuggingFace row format
evaluate   query one or more models on a dataset and write per-instance results
report     aggregate a run directory into ``summary.json`` and ``report.md``

The heavy lifting lives in the sibling modules; this file only parses
arguments. ``generate``/``solve`` are implemented in :mod:`generate` and
:mod:`solver`, ``evaluate``/``report`` in :mod:`evaluate` and :mod:`report`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .. import __version__
from .io import load_scenarios, to_hf_row, write_jsonl
from .prompts import final_message, round_message, system_prompt
from .rules import CRIMINAL, INVESTIGATOR, VARIANTS
from .scenario import Scenario

DEFAULT_MODELS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="socialmaze-hrd",
        description="SocialMaze Hidden Role Deduction: generate data, solve, evaluate models, report.",
    )
    parser.add_argument("--version", action="version", version=f"socialmaze {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    # -- generate ----------------------------------------------------------
    p = sub.add_parser("generate", help="generate a dataset of uniquely solvable games")
    p.add_argument("-n", "--num-players", type=int, default=6, help="number of players (default 6)")
    p.add_argument("--variant", default="full", choices=list(VARIANTS) + ["all"],
                   help="original | rumormonger | lunatic | full (default full; 'all' is an alias of full)")
    p.add_argument("--rumormongers", type=int, default=None,
                   help="number of Rumormongers (default: max(1, n//5) for variants that have them)")
    p.add_argument("--lunatics", type=int, default=None,
                   help="number of Lunatics (default: max(1, n//5) for variants that have them)")
    p.add_argument("--num-rounds", type=int, default=3, help="number of rounds (default 3)")
    p.add_argument("-N", "--num-scenarios", type=int, default=100,
                   help="number of uniquely solvable scenarios to keep (default 100)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    p.add_argument("--targeting", default="random", choices=["random", "strategic"],
                   help="statement policy: random (legacy-equivalent) or strategic (default random)")
    p.add_argument("--role-mix", default="uniform",
                   help="Player 1 role mix: 'uniform' (equal share of each role present), "
                        "'natural' (no rejection on role), or explicit weights such as "
                        "'Investigator=1,Criminal=1,Rumormonger=2,Lunatic=2' (default uniform)")
    p.add_argument("--max-attempts-factor", type=int, default=100,
                   help="give up after num_scenarios * factor simulated games (default 100)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSONL path (default data/hrd/hrd_n{n}_{variant}.jsonl)")
    p.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
    p.add_argument("--quiet", action="store_true", help="do not print progress")
    p.set_defaults(func=cmd_generate)

    # -- solve -------------------------------------------------------------
    p = sub.add_parser("solve", help="re-run the solver on a dataset and print uniqueness statistics")
    p.add_argument("data", type=Path, help="dataset file (.jsonl, legacy .json, or exported HF rows)")
    p.add_argument("--limit", type=int, default=None, help="only use the first N scenarios")
    p.add_argument("--explain", metavar="ID", default=None,
                   help="print the solver's reasoning chain for one scenario id")
    p.add_argument("--index", type=int, default=None,
                   help="with --explain: pick the scenario by 0-based index instead of id")
    p.add_argument("--json", action="store_true", help="print statistics as JSON")
    p.set_defaults(func=cmd_solve)

    # -- inspect -----------------------------------------------------------
    p = sub.add_parser("inspect", help="print one scenario in a readable form")
    p.add_argument("data", type=Path, help="dataset file")
    p.add_argument("--id", default=None, help="scenario id (default: first scenario)")
    p.add_argument("--index", type=int, default=0, help="0-based index if --id is not given")
    p.add_argument("--prompt", action="store_true",
                   help="also print the system prompt and user messages sent to a model")
    p.add_argument("--mode", default="incremental", choices=["incremental", "final"],
                   help="which user messages to render with --prompt (default incremental)")
    p.add_argument("--reasoning", action="store_true", help="also print the stored reasoning chain")
    p.set_defaults(func=cmd_inspect)

    # -- export ------------------------------------------------------------
    p = sub.add_parser("export", help="convert a dataset to the HuggingFace row format")
    p.add_argument("data", type=Path, help="dataset file")
    p.add_argument("--format", default="hf", choices=["hf"], help="target format (default hf)")
    p.add_argument("--out", type=Path, required=True,
                   help="output path (.jsonl for one row per line, .json for a list)")
    p.add_argument("--limit", type=int, default=None, help="only export the first N scenarios")
    p.set_defaults(func=cmd_export)

    # -- evaluate ----------------------------------------------------------
    p = sub.add_parser("evaluate", help="query models on a dataset and write per-instance results")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="dataset file (.jsonl, legacy .json, or HF rows)")
    src.add_argument("--from-hf", metavar="NAME", nargs="?", const="MBZUAI/SocialMaze",
                     help="load scenarios from the HuggingFace Hub (default MBZUAI/SocialMaze)")
    p.add_argument("--split", default="easy", help="HF split with --from-hf: easy (n=6) or hard (n=10)")
    p.add_argument("--models", nargs="+", required=True,
                   help="model names from configs/models.yaml, 'provider/model', or 'mock'")
    p.add_argument("--mode", default="incremental", choices=["incremental", "final"],
                   help="incremental: one Final Judgment after each round (paper protocol); "
                        "final: all rounds at once (default incremental)")
    p.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (default 0.7)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="completion cap per call (default: the model's max_tokens in the registry)")
    p.add_argument("--seeds", type=int, default=1, help="number of seeds 0..S-1 per scenario (default 1)")
    p.add_argument("--limit", type=int, default=None, help="only use the first N scenarios")
    p.add_argument("--workers", type=int, default=8, help="parallel requests per model (default 8)")
    p.add_argument("--out", type=Path, default=None,
                   help="run directory (default runs/<timestamp>); one <model>.jsonl per model")
    p.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG,
                   help="model registry YAML (default configs/models.yaml)")
    p.add_argument("--no-resume", action="store_true", help="ignore existing results in the run directory")
    p.add_argument("--no-retry-errors", action="store_true",
                   help="keep records whose rounds ended with an API error instead of re-querying them")
    p.add_argument("--quiet", action="store_true", help="do not print progress bars")
    p.set_defaults(func=cmd_evaluate)

    # -- report ------------------------------------------------------------
    p = sub.add_parser("report", help="aggregate a run directory into summary.json and report.md")
    p.add_argument("run_dir", type=Path, help="directory written by 'evaluate'")
    p.add_argument("--out", type=Path, default=None, help="markdown report path (default <run_dir>/report.md)")
    p.add_argument("--json-out", type=Path, default=None, help="summary path (default <run_dir>/summary.json)")
    p.add_argument("--quiet", action="store_true", help="do not print the report to stdout")
    p.set_defaults(func=cmd_report)

    return parser


# --------------------------------------------------------------------------
# Commands implemented in sibling modules
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    from .generate import run_generate

    return run_generate(args)


def cmd_solve(args: argparse.Namespace) -> int:
    from .solver import run_solve

    return run_solve(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_evaluate

    return run_evaluate(args)


def cmd_report(args: argparse.Namespace) -> int:
    from .report import run_report

    return run_report(args)


# --------------------------------------------------------------------------
# inspect / export
# --------------------------------------------------------------------------


def pick_scenario(scenarios: list[Scenario], sid: Optional[str], index: Optional[int]) -> Scenario:
    if sid is not None:
        for s in scenarios:
            if s.id == sid:
                return s
        raise SystemExit(f"no scenario with id {sid!r}")
    idx = index or 0
    if not 0 <= idx < len(scenarios):
        raise SystemExit(f"index {idx} out of range (dataset has {len(scenarios)} scenarios)")
    return scenarios[idx]


def format_scenario(sc: Scenario, show_prompt: bool = False, mode: str = "incremental",
                    show_reasoning: bool = False) -> str:
    lines = [
        f"id:             {sc.id}",
        f"variant:        {sc.variant} ({sc.num_players} players, {sc.num_rounds} rounds)",
        f"role counts:    {sc.config.role_counts}",
        f"Player 1 told:  {sc.displayed_role}",
        f"answer:         criminal = Player {sc.criminal}, Player 1 is {sc.player1_role}",
        "",
        "roles:",
    ]
    for p in sc.config.players:
        lines.append(f"  Player {p}: {sc.role_of(p) or '?'}")
    lines.append("")
    for t, rnd in enumerate(sc.rounds, start=1):
        lines.append(f"Round {t}:")
        for s in rnd:
            role = sc.role_of(s.speaker)
            truth = "true" if s.holds_for(sc.criminal) else "false"
            tag = f"[{role}, {truth}]" if role else f"[{truth}]"
            lines.append(f"  {s.render():<48} {tag}")
    if sc.solution:
        lines.append("")
        lines.append("solution:")
        for k, v in sc.solution.items():
            lines.append(f"  {k}: {v}")
    if sc.meta:
        lines.append("")
        lines.append(f"meta: {json.dumps(sc.meta, ensure_ascii=False)}")
    if show_reasoning:
        lines.append("")
        lines.append("reasoning:")
        lines.append(sc.reasoning or "(none)")
    if show_prompt:
        lines.append("")
        lines.append("=" * 20 + " system prompt " + "=" * 20)
        lines.append(system_prompt(sc.config, sc.displayed_role))
        if mode == "final":
            lines.append("=" * 20 + " user message " + "=" * 20)
            lines.append(final_message(sc.rounds))
        else:
            for t, rnd in enumerate(sc.rounds, start=1):
                lines.append("=" * 20 + f" user message, round {t} " + "=" * 20)
                lines.append(round_message(t, rnd))
    return "\n".join(lines)


def cmd_inspect(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.data)
    sc = pick_scenario(scenarios, args.id, args.index)
    print(format_scenario(sc, show_prompt=args.prompt, mode=args.mode, show_reasoning=args.reasoning))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.data, limit=args.limit)
    rows = [to_hf_row(s) for s in scenarios]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
    else:
        write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows in HuggingFace format to {out}")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
