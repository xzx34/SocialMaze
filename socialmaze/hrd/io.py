"""Reading and writing Hidden Role Deduction data.

Three formats are understood:

* **Native JSONL** (``data/hrd/*.jsonl``): one :class:`Scenario` per line,
  schema documented in :mod:`socialmaze.hrd.scenario`. A dataset may carry a
  sidecar ``<stem>.meta.json`` with generation settings and statistics.
* **Legacy JSON** (``archive/hidden_role_deduction/data/hrd_{n}_{type}.json``):
  the list-of-dicts format of the original release. Converted on load.
* **HuggingFace rows** (``MBZUAI/SocialMaze``): flat rows with ``task``,
  ``system_prompt``, ``prompt``, ``answer``, ``reasoning_process`` and
  ``round 1`` .. ``round 3`` columns. Converted on load, and produced by
  :func:`to_hf_row` for export. Only Player 1's role and the Criminal are
  recoverable from such rows, so ``Scenario.roles`` is partial for them.

:func:`load_scenarios` dispatches on file extension and record shape, so every
command accepts any of the three formats.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from .prompts import answer_block, final_message, round_block, system_prompt
from .rules import (
    CRIMINAL,
    DISPLAYED_ROLE,
    INVESTIGATOR,
    PLAYER_ONE,
    ROLES,
    GameConfig,
    Statement,
    normalize_variant,
)
from .scenario import TASK_NAME, Answer, Scenario

PathLike = Union[str, Path]

# --------------------------------------------------------------------------
# JSONL primitives
# --------------------------------------------------------------------------


def iter_jsonl(path: PathLike) -> Iterator[dict]:
    """Yield one decoded object per non-empty line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: PathLike) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: PathLike, records: Iterable[dict]) -> int:
    """Write ``records`` one per line (UTF-8, no ASCII escaping). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: PathLike, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def meta_path(path: PathLike) -> Path:
    """Sidecar path: ``hrd_n6_full.jsonl`` -> ``hrd_n6_full.meta.json``."""
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def load_meta(path: PathLike) -> Optional[dict]:
    """Return the sidecar metadata of a dataset file, or ``None``."""
    mp = meta_path(path)
    if not mp.exists():
        return None
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def save_scenarios(
    path: PathLike, scenarios: Iterable[Scenario], meta: Optional[dict] = None
) -> Path:
    """Write scenarios as native JSONL and, if given, the ``.meta.json`` sidecar."""
    path = Path(path)
    write_jsonl(path, (s.to_dict() for s in scenarios))
    if meta is not None:
        with open(meta_path(path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return path


# --------------------------------------------------------------------------
# Legacy JSON (original release)
# --------------------------------------------------------------------------


def is_legacy_record(d: dict) -> bool:
    return "scenario_id" in d and "statements" in d and "ground_truth" in d


def from_legacy_record(rec: dict, index: Optional[int] = None) -> Scenario:
    """Convert one record of ``archive/hidden_role_deduction/data/*.json``."""
    roles = {int(p): r for p, r in rec["roles"].items()}
    counts = {r: 0 for r in ROLES}
    for r in roles.values():
        counts[r] += 1
    rounds_raw = sorted(rec["statements"], key=lambda r: int(r["round"]))
    cfg = GameConfig.from_role_counts(counts, num_rounds=len(rounds_raw))
    rounds = [
        [
            Statement(int(s["player"]), int(s["target_player"]), s["statement_type"])
            for s in sorted(r["statements"], key=lambda s: int(s["player"]))
        ]
        for r in rounds_raw
    ]
    gt = rec["ground_truth"]
    answer = Answer(int(gt["criminal"]), gt["player1_role"])
    analysis = rec.get("solution_analysis") or {}
    dataset_type = rec.get("dataset_type")
    variant = normalize_variant(dataset_type) if dataset_type else cfg.variant
    sid = rec.get("scenario_id", index)
    return Scenario(
        id=f"hrd-n{cfg.num_players}-{variant}-legacy-{sid}",
        config=cfg,
        displayed_role=DISPLAYED_ROLE[roles[PLAYER_ONE]],
        rounds=rounds,
        answer=answer,
        roles=roles,
        solution=None,
        reasoning=analysis.get("reasoning_process"),
        meta={
            "source": "legacy",
            "legacy_dataset_type": dataset_type,
            "legacy_unique_solution": analysis.get("unique_solution"),
        },
    )


# --------------------------------------------------------------------------
# HuggingFace rows (MBZUAI/SocialMaze)
# --------------------------------------------------------------------------

_PLAYERS_RE = re.compile(r"(\d+)\s+players", re.IGNORECASE)
_ROUNDS_RE = re.compile(r"(\d+)\s+rounds", re.IGNORECASE)
_TOLD_RE = re.compile(
    r"told\s+that\s+you\s+are\s+(?:an?\s+|the\s+)?(Investigator|Criminal)", re.IGNORECASE
)
_ANSWER_CRIMINAL_RE = re.compile(r"Final\s+Criminal\s+Is\s+Player\s*#?\s*(\d+)", re.IGNORECASE)
_ANSWER_ROLE_RE = re.compile(r"My\s+Role\s+Is\s*[*\"'`]*\s*([A-Za-z]+)", re.IGNORECASE)
_ROUND_KEY_RE = re.compile(r"^round\s*(\d+)$", re.IGNORECASE)


def parse_config_from_system_prompt(text: str) -> tuple[GameConfig, str]:
    """Recover ``(GameConfig, displayed_role)`` from a system prompt.

    Works for both the prompt of the original release and the current one
    (:func:`socialmaze.hrd.prompts.system_prompt`).
    """
    counts = {}
    for role in ROLES:
        m = re.search(rf"(\d+)\s+{role}", text)
        counts[role] = int(m.group(1)) if m else 0
    m_players = _PLAYERS_RE.search(text)
    m_rounds = _ROUNDS_RE.search(text)
    m_told = _TOLD_RE.search(text)
    if not (m_players and m_told):
        raise ValueError("could not parse player count or displayed role from system prompt")
    num_rounds = int(m_rounds.group(1)) if m_rounds else 3
    cfg = GameConfig.from_role_counts(counts, num_rounds=num_rounds)
    if cfg.num_players != int(m_players.group(1)):
        raise ValueError("role counts in system prompt do not add up to the player count")
    told = m_told.group(1).capitalize()
    return cfg, told


def parse_answer_text(text: str) -> Answer:
    """Parse ``"Final Criminal Is Player 4.\\nMy Role Is Lunatic."``."""
    mc = _ANSWER_CRIMINAL_RE.search(text)
    mr = _ANSWER_ROLE_RE.search(text)
    if not (mc and mr):
        raise ValueError(f"could not parse answer text {text!r}")
    role = mr.group(1).capitalize()
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r} in answer text")
    return Answer(int(mc.group(1)), role)


def is_hf_row(d: dict) -> bool:
    return "system_prompt" in d and "answer" in d and any(_ROUND_KEY_RE.match(k) for k in d)


def from_hf_row(row: dict, index: Optional[int] = None, id_prefix: str = "hf") -> Scenario:
    """Convert one row of the HuggingFace dataset (or an exported HF-format row)."""
    cfg, displayed = parse_config_from_system_prompt(row["system_prompt"])
    round_keys = sorted(
        (int(_ROUND_KEY_RE.match(k).group(1)), k) for k in row if _ROUND_KEY_RE.match(k)
    )
    rounds = [Statement.parse_all(row[k] or "") for _, k in round_keys]
    rounds = [r for r in rounds if r]
    if len(rounds) != cfg.num_rounds:
        cfg = GameConfig(cfg.num_players, cfg.num_rumormongers, cfg.num_lunatics, len(rounds))
    answer = parse_answer_text(row["answer"])
    roles = {PLAYER_ONE: answer.player1_role, answer.criminal: CRIMINAL}
    sid = row.get("id") or f"{id_prefix}-{index if index is not None else 0:06d}"
    return Scenario(
        id=str(sid),
        config=cfg,
        displayed_role=displayed,
        rounds=rounds,
        answer=answer,
        roles=roles,
        solution=None,
        reasoning=row.get("reasoning_process") or None,
        meta={"source": "huggingface"},
    )


def to_hf_row(scenario: Scenario) -> dict:
    """Render a scenario in the flat row format of the HuggingFace release."""
    row = {
        "task": TASK_NAME,
        "system_prompt": system_prompt(scenario.config, scenario.displayed_role),
        "prompt": final_message(scenario.rounds),
        "answer": answer_block(scenario.criminal, scenario.player1_role, header=False),
        "reasoning_process": scenario.reasoning or "",
    }
    for t, rnd in enumerate(scenario.rounds, start=1):
        row[f"round {t}"] = round_block(t, rnd)
    row["id"] = scenario.id
    return row


def load_hf(
    name: str = "MBZUAI/SocialMaze",
    split: str = "easy",
    limit: Optional[int] = None,
    streaming: bool = True,
) -> list[Scenario]:
    """Load scenarios straight from the HuggingFace Hub (needs the ``datasets`` extra).

    ``easy`` is the six-player full variant and ``hard`` the ten-player full
    variant of the original release.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "loading from the HuggingFace Hub requires the 'datasets' package: "
            "pip install 'socialmaze[hf]'"
        ) from exc
    ds = load_dataset(name, split=split, streaming=streaming)
    out: list[Scenario] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(from_hf_row(row, index=i, id_prefix=f"hf-{split}"))
    return out


# --------------------------------------------------------------------------
# Generic loading
# --------------------------------------------------------------------------


def record_to_scenario(d: dict, index: Optional[int] = None) -> Scenario:
    """Convert a record of any supported shape into a :class:`Scenario`."""
    if "rounds" in d and "answer" in d:
        return Scenario.from_dict(d)
    if is_legacy_record(d):
        return from_legacy_record(d, index)
    if is_hf_row(d):
        return from_hf_row(d, index)
    raise ValueError("unrecognised scenario record")


def load_scenarios(path: PathLike, limit: Optional[int] = None) -> list[Scenario]:
    """Load scenarios from a ``.jsonl`` or ``.json`` file of any supported format."""
    path = Path(path)
    if path.suffix == ".jsonl":
        records: Iterable[dict] = iter_jsonl(path)
    elif path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data") or data.get("scenarios") or [data]
        records = data
    else:
        raise ValueError(f"unsupported file type: {path.suffix!r} (expected .jsonl or .json)")
    out: list[Scenario] = []
    for i, rec in enumerate(records):
        if limit is not None and len(out) >= limit:
            break
        out.append(record_to_scenario(rec, i))
    return out


__all__ = [
    "iter_jsonl", "read_jsonl", "write_jsonl", "append_jsonl", "meta_path",
    "load_meta", "save_scenarios", "is_legacy_record", "from_legacy_record",
    "parse_config_from_system_prompt", "parse_answer_text", "is_hf_row",
    "from_hf_row", "to_hf_row", "load_hf", "record_to_scenario", "load_scenarios",
]
