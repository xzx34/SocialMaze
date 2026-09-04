"""Aggregation of a run directory into summary.json and report.md."""

import argparse
import json
from pathlib import Path

import pytest

from socialmaze.hrd.evaluate import evaluate, run_manifest
from socialmaze.hrd.io import load_scenarios
from socialmaze.hrd.report import (
    format_pct,
    load_run,
    markdown_table,
    ordered_models,
    render_markdown,
    run_report,
    summarize_run,
    write_report,
)
from socialmaze.llm.client import ModelSpec
from socialmaze.llm.mock import MockClient

LEGACY = Path(__file__).resolve().parents[1] / "archive" / "hidden_role_deduction" / "data" / "hrd_6_all.json"
MODELS = {"mock-oracle": "oracle", "mock-wrong": "wrong", "mock-garbage": "garbage"}


def make_run(run_dir, seeds=1, models=MODELS):
    scenarios = load_scenarios(LEGACY)
    specs = []
    for name, behaviour in models.items():
        spec = ModelSpec(name=name, provider="mock", model=behaviour)
        specs.append(spec)
        evaluate(scenarios, spec, seeds=seeds, out_path=run_dir / f"{name}.jsonl",
                 client=MockClient(behaviour), progress=False, workers=4)
    args = argparse.Namespace(data=LEGACY, from_hf=None, split="easy", limit=None, mode="incremental",
                              temperature=0.7, max_tokens=None, seeds=seeds)
    (run_dir / "run.json").write_text(json.dumps(run_manifest(args, specs, scenarios, None)))
    return scenarios


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("run")
    make_run(path)
    return path


def test_load_run_groups_by_model_field(run_dir):
    by_model = load_run(run_dir)
    assert set(by_model) == set(MODELS)
    assert all(len(records) == 10 for records in by_model.values())
    assert all(r["model"] == name for name, records in by_model.items() for r in records)
    with pytest.raises(FileNotFoundError):
        load_run(run_dir / "missing")


def test_ordered_models_follows_manifest():
    assert ordered_models(["c", "a", "b"], {"models": ["b", "c"]}) == ["b", "c", "a"]
    assert ordered_models(["c", "a"], {}) == ["a", "c"]


def test_summarize_run(run_dir):
    summary = summarize_run(run_dir)
    assert summary["run"]["mode"] == "incremental" and summary["run"]["num_scenarios"] == 10
    assert list(summary["models"]) == list(MODELS)
    oracle = summary["models"]["mock-oracle"]
    assert oracle["rounds"]["3"]["both_acc"] == 1.0 and oracle["num_records"] == 10
    assert summary["models"]["mock-garbage"]["errors"]["fractions"]["extraction_failed"] == 1.0
    assert set(oracle["by_role"]) == {"Lunatic", "Rumormonger"}
    json.dumps(summary)


def test_render_markdown_has_every_table(run_dir):
    text = render_markdown(summarize_run(run_dir))
    for name in MODELS:
        assert name in text
        for t in (1, 2, 3):
            assert f"| {name} | {t} | 10 |" in text
    assert "## Accuracy per round" in text
    assert "| Model | Round | n | Crim. | Self | Both | Unknown |" in text
    assert "## Accuracy at the final round by Player 1 true role" in text
    assert "| mock-oracle | Lunatic | 5 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |" in text
    assert "| mock-oracle | Rumormonger | 5 |" in text
    assert "## Error decomposition at the final round" in text
    assert "| Model | n | API error | Truncated | No judgment | Hedged | Reasoning error | Correct |" in text
    assert "| mock-garbage | 10 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |" in text
    assert "| mock-wrong | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |" in text
    assert "## Usage at the final round" in text
    assert "- data: " in text and "- seeds: 1" in text
    assert "Variation across seeds" not in text


def test_render_markdown_with_seeds(tmp_path):
    make_run(tmp_path, seeds=2, models={"mock-oracle": "oracle"})
    summary = summarize_run(tmp_path)
    assert summary["models"]["mock-oracle"]["num_seeds"] == 2
    text = render_markdown(summary)
    assert "## Variation across seeds at the final round" in text
    assert "| mock-oracle | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |" in text
    assert "| mock-oracle | 3 | 20 |" in text


def test_render_markdown_without_manifest(tmp_path):
    make_run(tmp_path, models={"mock-oracle": "oracle"})
    (tmp_path / "run.json").unlink()
    text = render_markdown(summarize_run(tmp_path))
    assert "settings unknown" in text and "mock-oracle" in text


def test_write_report_and_run_report(run_dir, tmp_path, capsys):
    md_path, json_path = write_report(run_dir)
    assert md_path == run_dir / "report.md" and json_path == run_dir / "summary.json"
    assert md_path.exists() and json_path.exists()
    stored = json.loads(json_path.read_text())
    assert set(stored["models"]) == set(MODELS)
    assert md_path.read_text() == render_markdown(stored)

    custom_md, custom_json = tmp_path / "out" / "r.md", tmp_path / "out" / "s.json"
    args = argparse.Namespace(run_dir=run_dir, out=custom_md, json_out=custom_json, quiet=False)
    assert run_report(args) == 0
    assert custom_md.exists() and custom_json.exists()
    out = capsys.readouterr().out
    assert "# SocialMaze" in out and f"wrote {custom_md}" in out
    assert run_report(argparse.Namespace(run_dir=run_dir, out=None, json_out=None, quiet=True)) == 0
    assert "# SocialMaze" not in capsys.readouterr().out


def test_formatting_helpers():
    assert format_pct(0.4123, 0.0431) == "41.2 ± 4.3"
    assert format_pct(1.0) == "100.0"
    assert format_pct(None) == "-"
    assert markdown_table(["a", "b"], [[1, "x"]]) == "| a | b |\n|---|---|\n| 1 | x |"
