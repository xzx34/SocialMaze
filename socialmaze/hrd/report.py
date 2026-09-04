"""Aggregate a run directory into ``summary.json`` and ``report.md``.

A run directory is written by :mod:`socialmaze.hrd.evaluate`: ``run.json``
with the settings, and one ``<model>.jsonl`` of result records per model.
:func:`summarize_run` applies :func:`socialmaze.hrd.metrics.summarize_records`
to every model and :func:`render_markdown` turns the summaries into the
tables of the paper:

1. accuracy per model and round (Crim., Self, Both as ``mean ± half-width``
   percentages, plus the Unknown rate);
2. accuracy at the final round per Player 1 true role;
3. the error decomposition at the final round;
4. mean prompt and completion tokens and latency at the final round;
5. when a model was run with several seeds, the mean and sample standard
   deviation of its final-round accuracies across seeds.

Records are grouped by the ``model`` field of each record, so a file may be
renamed without affecting the report. Models are listed in the order of
``run.json`` when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Union

from .io import read_jsonl
from .metrics import ACCURACIES, ERROR_CATEGORIES, summarize_records

RUN_FILE = "run.json"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"

ERROR_LABELS: dict[str, str] = {
    "api_error": "API error",
    "truncated": "Truncated",
    "extraction_failed": "No judgment",
    "hedged": "Hedged",
    "reasoning_error": "Reasoning error",
    "correct": "Correct",
}

PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_manifest(run_dir: PathLike) -> dict:
    """The ``run.json`` contents, or an empty dict if the file is absent."""
    path = Path(run_dir) / RUN_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir: PathLike) -> dict[str, list[dict]]:
    """Result records per model name from every ``*.jsonl`` file in ``run_dir``."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist")
    by_model: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        for record in read_jsonl(path):
            by_model.setdefault(record.get("model") or path.stem, []).append(record)
    return by_model


def ordered_models(names: Sequence[str], manifest: dict) -> list[str]:
    """``names`` in ``run.json`` order first, then the rest alphabetically."""
    listed = [m for m in manifest.get("models", []) if m in names]
    return listed + sorted(n for n in names if n not in listed)


def summarize_run(run_dir: PathLike) -> dict:
    """``{"run": run.json contents, "models": {name: summary}}``."""
    manifest = load_manifest(run_dir)
    by_model = load_run(run_dir)
    return {
        "run": manifest,
        "models": {name: summarize_records(by_model[name]) for name in ordered_models(list(by_model), manifest)},
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def format_pct(p: Optional[float], halfwidth: Optional[float] = None) -> str:
    """``62.4`` or ``62.4 ± 4.2`` (percent, one decimal)."""
    if p is None:
        return "-"
    if halfwidth is None:
        return f"{100 * p:.1f}"
    return f"{100 * p:.1f} ± {100 * halfwidth:.1f}"


def format_num(value: Optional[float], decimals: int = 1) -> str:
    return "-" if value is None else f"{value:.{decimals}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _accuracy_cells(stats: dict) -> list[str]:
    return [format_pct(stats[f"{p}_acc"], stats[f"{p}_halfwidth"]) for p, _ in ACCURACIES]


def render_settings(run: dict) -> str:
    if not run:
        return "No run.json found; settings unknown."
    source = run.get("data") or (f"{run.get('from_hf')} ({run.get('split')})" if run.get("from_hf") else "?")
    items = [
        f"data: {source}",
        f"scenarios: {run.get('num_scenarios', '?')}",
        f"mode: {run.get('mode', '?')}",
        f"temperature: {run.get('temperature', '?')}",
        f"max_tokens: {run.get('max_tokens') or 'registry default'}",
        f"seeds: {run.get('seeds', '?')}",
        f"created: {run.get('created', '?')}",
        f"socialmaze: {run.get('socialmaze_version', '?')}",
    ]
    return "\n".join(f"- {item}" for item in items)


def render_rounds_table(models: dict) -> str:
    rows = []
    for name, s in models.items():
        for t, stats in s["rounds"].items():
            rows.append([name, t, stats["n"], *_accuracy_cells(stats), format_pct(stats["unknown_rate"])])
    return markdown_table(["Model", "Round", "n", "Crim.", "Self", "Both", "Unknown"], rows)


def render_roles_table(models: dict) -> str:
    rows = []
    for name, s in models.items():
        for role, stats in s["by_role"].items():
            rows.append([name, role, stats["n"], *_accuracy_cells(stats)])
    return markdown_table(["Model", "Player 1 role", "n", "Crim.", "Self", "Both"], rows)


def render_errors_table(models: dict) -> str:
    rows = []
    for name, s in models.items():
        fractions = s["errors"]["fractions"]
        rows.append([name, s["errors"]["n"], *[format_pct(fractions[c]) for c in ERROR_CATEGORIES]])
    return markdown_table(["Model", "n", *[ERROR_LABELS[c] for c in ERROR_CATEGORIES]], rows)


def render_usage_table(models: dict) -> str:
    rows = []
    for name, s in models.items():
        u = s["usage"]
        rows.append([
            name, u["n"], format_num(u["prompt_tokens_mean"], 0), format_num(u["completion_tokens_mean"], 0),
            format_num(u["latency_s_mean"], 2), u["prompt_tokens_total"], u["completion_tokens_total"],
        ])
    return markdown_table(
        ["Model", "n", "Prompt tokens", "Completion tokens", "Latency (s)", "Prompt total", "Completion total"], rows
    )


def render_seeds_table(models: dict) -> str:
    rows = []
    for name, s in models.items():
        if s["num_seeds"] > 1:
            cells = [f"{format_pct(s['seed_mean'][k])} ± {format_pct(s['seed_std'][k])}" for k in ("crim_acc", "self_acc", "both_acc")]
            rows.append([name, s["num_seeds"], *cells])
    return markdown_table(["Model", "Seeds", "Crim. (mean ± std)", "Self (mean ± std)", "Both (mean ± std)"], rows)


def render_markdown(summary: dict) -> str:
    """The full report; see the module docstring for the tables."""
    models = summary["models"]
    sections = [
        "# SocialMaze Hidden Role Deduction: evaluation report",
        "",
        "## Settings",
        "",
        render_settings(summary.get("run") or {}),
        "",
        "## Accuracy per round",
        "",
        "Percent with 95% binomial half-width. Crim.: Criminal identified; Self: own role "
        "(strict); Both: both correct; Unknown: share of replies answering \"Unknown\" for the role.",
        "",
        render_rounds_table(models),
        "",
        "## Accuracy at the final round by Player 1 true role",
        "",
        render_roles_table(models),
        "",
        "## Error decomposition at the final round",
        "",
        "Percent of records per mutually exclusive category: the call failed (API error), the "
        "reply hit the completion cap (Truncated), no criminal line (No judgment), two roles on "
        "the role line (Hedged), a well-formed but wrong answer (Reasoning error), or Correct.",
        "",
        render_errors_table(models),
        "",
        "## Usage at the final round",
        "",
        render_usage_table(models),
    ]
    if any(s["num_seeds"] > 1 for s in models.values()):
        sections += [
            "",
            "## Variation across seeds at the final round",
            "",
            "Mean and sample standard deviation of the per-seed accuracies, in percent.",
            "",
            render_seeds_table(models),
        ]
    return "\n".join(sections) + "\n"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_report(
    run_dir: PathLike,
    md_path: Optional[PathLike] = None,
    json_path: Optional[PathLike] = None,
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``summary.json`` into ``run_dir``; returns ``(md, json)`` paths."""
    run_dir = Path(run_dir)
    md_path = Path(md_path) if md_path else run_dir / REPORT_FILE
    json_path = Path(json_path) if json_path else run_dir / SUMMARY_FILE
    summary = summarize_run(run_dir)
    for path in (md_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return md_path, json_path


def run_report(args: argparse.Namespace) -> int:
    """The ``report`` command; see :func:`socialmaze.hrd.cli.build_parser`."""
    md_path, json_path = write_report(args.run_dir, args.out, args.json_out)
    if not args.quiet:
        print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {md_path} and {json_path}")
    return 0


__all__ = [
    "RUN_FILE", "SUMMARY_FILE", "REPORT_FILE", "ERROR_LABELS", "load_manifest",
    "load_run", "ordered_models", "summarize_run", "format_pct", "format_num",
    "markdown_table", "render_markdown", "write_report", "run_report",
]
