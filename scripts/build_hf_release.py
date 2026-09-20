#!/usr/bin/env python3
"""Build and fully validate the SocialMaze HRD Hugging Face release."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq

from socialmaze.hrd.generate import content_fingerprint, generate_dataset
from socialmaze.hrd.io import from_hf_row, to_hf_row
from socialmaze.hrd.prompts import answer_block
from socialmaze.hrd.rules import ROLES, GameConfig
from socialmaze.hrd.solver import analyze

VERSION = "2.0.0"
DEFAULT_SEED = 20260920
SCHEMA = [
    "task", "system_prompt", "prompt", "answer", "reasoning_process",
    "round 1", "round 2", "round 3", "id",
]
SPLITS = {
    "easy": {"players": 6, "shards": 2},
    "hard": {"players": 10, "shards": 5},
}


def canonical_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_split(name: str, scenarios: list, rows: list[dict], expected: int) -> dict:
    errors: list[str] = []
    ids: set[str] = set()
    fingerprints: set[str] = set()
    roles = Counter()
    for index, (scenario, row) in enumerate(zip(scenarios, rows), start=1):
        try:
            scenario.validate()
            if list(row) != SCHEMA:
                raise ValueError(f"schema/order is {list(row)!r}")
            if scenario.id in ids:
                raise ValueError("duplicate id")
            ids.add(scenario.id)
            fingerprint = content_fingerprint(scenario)
            if fingerprint in fingerprints:
                raise ValueError("duplicate observable content")
            fingerprints.add(fingerprint)
            solution = analyze(scenario)
            if not solution["unique"]:
                raise ValueError("solver found ambiguity")
            if (solution["criminal"], solution["player1_role"]) != (
                scenario.criminal, scenario.player1_role
            ):
                raise ValueError("stored answer disagrees with solver")
            if solution != scenario.solution:
                raise ValueError("stored solver record disagrees with recomputation")
            if not scenario.reasoning.endswith(answer_block(scenario.criminal, scenario.player1_role)):
                raise ValueError("reasoning does not terminate in the stored answer")
            restored = from_hf_row(row)
            if (
                restored.id != scenario.id
                or restored.config != scenario.config
                or restored.displayed_role != scenario.displayed_role
                or restored.rounds != scenario.rounds
                or restored.answer != scenario.answer
                or restored.reasoning != scenario.reasoning
            ):
                raise ValueError("HF row round-trip mismatch")
            roles[scenario.player1_role] += 1
        except Exception as exc:  # collect a useful bounded report
            if len(errors) < 100:
                errors.append(f"{name}[{index}]: {exc}")
    target_per_role = expected // len(ROLES)
    expected_roles = {role: target_per_role for role in ROLES}
    if len(scenarios) != expected or len(rows) != expected:
        errors.append(f"expected {expected} rows, got {len(scenarios)} scenarios/{len(rows)} rows")
    if dict(roles) != expected_roles:
        errors.append(f"role distribution {dict(roles)!r}, expected {expected_roles!r}")
    if len(ids) != expected:
        errors.append(f"only {len(ids)} unique ids")
    if len(fingerprints) != expected:
        errors.append(f"only {len(fingerprints)} unique content fingerprints")
    return {
        "status": "passed" if not errors else "failed",
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_content_fingerprints": len(fingerprints),
        "player1_role_counts": dict(roles),
        "checks": [
            "corrected exhaustive solver unique",
            "stored answer matches solver",
            "stored solver record matches recomputation",
            "stable id unique within split",
            "observable content fingerprint unique within split",
            "schema and column order",
            "HF row round-trip",
            "reasoning terminates in stored answer",
            "exact Player 1 role balance",
        ],
        "errors": errors,
    }


def write_shards(rows: list[dict], output: Path, split: str, count: int) -> list[str]:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shard_size = math.ceil(len(rows) / count)
    names = []
    for shard_index in range(count):
        chunk = rows[shard_index * shard_size:(shard_index + 1) * shard_size]
        name = f"{split}-{shard_index:05d}-of-{count:05d}.parquet"
        table = pa.Table.from_pylist(chunk, schema=pa.schema([(column, pa.string()) for column in SCHEMA]))
        pq.write_table(table, data_dir / name, compression="zstd")
        names.append(f"data/{name}")
    return names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-split", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-determinism-check", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.rows_per_split <= 0 or args.rows_per_split % len(ROLES):
        parser.error("--rows-per-split must be positive and divisible by four")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "hf_dataset" / "README.md", output / "README.md")
    shutil.copy2(REPO_ROOT / "hf_dataset" / "VERSION", output / "VERSION")

    report = {
        "release": VERSION,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "splits": {},
    }
    for split, spec in SPLITS.items():
        config = GameConfig.create(spec["players"], "full")
        scenarios, stats = generate_dataset(
            config,
            args.rows_per_split,
            role_mix="uniform",
            targeting="random",
            seed=args.seed,
            progress=args.progress,
            id_prefix=f"socialmaze-v2-{split}",
        )
        rows = [to_hf_row(scenario) for scenario in scenarios]
        validation = validate_split(split, scenarios, rows, args.rows_per_split)
        content_hash = canonical_hash(rows)
        deterministic_hash = None
        if not args.skip_determinism_check:
            second, second_stats = generate_dataset(
                config,
                args.rows_per_split,
                role_mix="uniform",
                targeting="random",
                seed=args.seed,
                progress=args.progress,
                id_prefix=f"socialmaze-v2-{split}",
            )
            deterministic_hash = canonical_hash([to_hf_row(scenario) for scenario in second])
            if deterministic_hash != content_hash or second_stats != stats:
                validation["status"] = "failed"
                validation["errors"].append("same-seed full regeneration did not match")
            del second
            gc.collect()
        validation["deterministic_regeneration"] = (
            "passed" if deterministic_hash == content_hash else
            "skipped" if args.skip_determinism_check else "failed"
        )
        validation["canonical_rows_sha256"] = content_hash
        files = write_shards(rows, output, split, spec["shards"])
        metadata = {
            "release": VERSION,
            "split": split,
            "seed": args.seed,
            "config": config.to_dict(),
            "rows": args.rows_per_split,
            "parquet_shards": files,
            "canonical_rows_sha256": content_hash,
            "generator_statistics": stats,
        }
        (output / f"generation_metadata_{split}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report["splits"][split] = validation
        del scenarios, rows
        gc.collect()

    report["status"] = (
        "passed" if all(item["status"] == "passed" for item in report["splits"].values()) else "failed"
    )
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    manifest = "".join(f"{sha256_file(path)}  {path.relative_to(output)}\n" for path in files)
    (output / "checksums.sha256").write_text(manifest, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
