#!/usr/bin/env python3
"""Tag the legacy dataset, publish v2, and verify the Hub copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_config_info, load_dataset
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--repo", default="xzx34/SocialMaze")
    args = parser.parse_args()
    release = args.release.resolve()
    report = json.loads((release / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        parser.error("validation_report.json is not passed")
    api = HfApi()
    api.whoami()
    refs = api.list_repo_refs(args.repo, repo_type="dataset")
    tag_names = {tag.name for tag in refs.tags}
    if "legacy-v1" not in tag_names:
        api.create_tag(
            args.repo,
            repo_type="dataset",
            tag="legacy-v1",
            revision="main",
            tag_message="Preserve the pre-v2 2025 personal release",
        )
    local_files = {str(path.relative_to(release)) for path in release.rglob("*") if path.is_file()}
    remote_files = set(api.list_repo_files(args.repo, repo_type="dataset", revision="main"))
    stale_files = sorted(remote_files - local_files - {".gitattributes"})
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=release,
        revision="main",
        delete_patterns=stale_files or None,
        commit_message="Release corrected expanded HRD dataset v2.0.0",
        commit_description=(
            "200,000 exhaustive-solver-validated rows; fixed seed 20260920; "
            "uniform Player 1 roles; stable IDs; checksums and validation report."
        ),
    )
    files = set(api.list_repo_files(args.repo, repo_type="dataset", revision="main"))
    missing = local_files - files
    if missing:
        raise RuntimeError(f"uploaded repository is missing: {sorted(missing)}")
    info = get_dataset_config_info(args.repo)
    counts = {
        name: split.get("num_examples") if isinstance(split, dict) else split.num_examples
        for name, split in info.splits.items()
    }
    online_dataset = None
    if any(value is None for value in counts.values()):
        online_dataset = load_dataset(args.repo)
        counts = {name: len(split) for name, split in online_dataset.items()}
    if counts != {"easy": 100_000, "hard": 100_000}:
        raise RuntimeError(f"unexpected online split counts: {counts}")
    for split in ("easy", "hard"):
        first = (
            online_dataset[split][0]
            if online_dataset is not None
            else next(iter(load_dataset(args.repo, split=split, streaming=True)))
        )
        if not first.get("id", "").startswith(f"socialmaze-v2-{split}-"):
            raise RuntimeError(f"unexpected online {split} row id: {first.get('id')!r}")
    print(json.dumps({"repo": args.repo, "tag": "legacy-v1", "splits": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
