"""The Scenario record: one Hidden Role Deduction game instance.

A scenario bundles the game configuration, the (hidden) role of every player,
the role Player 1 was told, the public statements of every round, the ground
truth answer, and optionally the solver's verdict and a natural-language
reasoning chain. Scenarios are stored one per line in JSONL files (see
:mod:`socialmaze.hrd.io`). The on-disk schema is::

    {
      "id": "hrd-n6-full-00001",
      "task": "Hidden Role Deduction",
      "num_players": 6, "num_rounds": 3, "variant": "full",
      "role_counts": {"Investigator": 3, "Criminal": 1, "Rumormonger": 1, "Lunatic": 1},
      "roles": {"1": "Rumormonger", "2": "Investigator", ...},
      "displayed_role": "Investigator",
      "rounds": [[{"speaker": 1, "target": 4, "claim": "is_not"}, ...], ...],
      "answer": {"criminal": 4, "player1_role": "Rumormonger"},
      "solution": {"unique": true, "solvable_after_round": 2, ...} | null,
      "reasoning": "..." | null,
      "meta": {"generator_version": "1.0.0", "targeting": "random", "seed": 0}
    }

``roles`` is complete for generated data. For scenarios imported from the
HuggingFace release only Player 1's role and the Criminal are known, so
``roles`` is partial there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .rules import (
    CRIMINAL,
    DISPLAYED_ROLE,
    DISPLAYED_ROLES,
    PLAYER_ONE,
    ROLES,
    GameConfig,
    Statement,
)

TASK_NAME = "Hidden Role Deduction"


@dataclass
class Answer:
    """Ground truth: who the Criminal is and Player 1's true role."""

    criminal: int
    player1_role: str

    def to_dict(self) -> dict:
        return {"criminal": self.criminal, "player1_role": self.player1_role}

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        return cls(int(d["criminal"]), str(d["player1_role"]))


@dataclass
class Scenario:
    """One game instance seen from Player 1's perspective."""

    id: str
    config: GameConfig
    displayed_role: str
    rounds: list[list[Statement]]
    answer: Answer
    roles: dict[int, str] = field(default_factory=dict)
    solution: Optional[dict] = None
    reasoning: Optional[str] = None
    meta: dict = field(default_factory=dict)
    task: str = TASK_NAME

    # -- convenience -------------------------------------------------------

    @property
    def num_players(self) -> int:
        return self.config.num_players

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def variant(self) -> str:
        return self.config.variant

    @property
    def criminal(self) -> int:
        return self.answer.criminal

    @property
    def player1_role(self) -> str:
        return self.answer.player1_role

    def statements_through(self, round_index: int) -> list[Statement]:
        """All statements of rounds ``1..round_index`` (1-based, inclusive)."""
        out: list[Statement] = []
        for r in self.rounds[:round_index]:
            out.extend(r)
        return out

    def all_statements(self) -> list[Statement]:
        return self.statements_through(len(self.rounds))

    def role_of(self, player: int) -> Optional[str]:
        """True role of ``player`` if known, else ``None``."""
        return self.roles.get(int(player))

    def has_complete_roles(self) -> bool:
        return len(self.roles) == self.num_players

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if the record is internally inconsistent."""
        n = self.config.num_players
        if self.displayed_role not in DISPLAYED_ROLES:
            raise ValueError(f"{self.id}: bad displayed_role {self.displayed_role!r}")
        if self.answer.player1_role not in ROLES:
            raise ValueError(f"{self.id}: bad player1_role {self.answer.player1_role!r}")
        if DISPLAYED_ROLE[self.answer.player1_role] != self.displayed_role:
            raise ValueError(f"{self.id}: player1_role does not match displayed_role")
        if self.answer.player1_role not in self.config.true_roles_for(self.displayed_role):
            raise ValueError(f"{self.id}: player1_role not present in this variant")
        if not 1 <= self.answer.criminal <= n:
            raise ValueError(f"{self.id}: criminal out of range")
        if self.answer.player1_role == CRIMINAL and self.answer.criminal != PLAYER_ONE:
            raise ValueError(f"{self.id}: Player 1 is the Criminal but criminal != 1")
        if self.answer.player1_role != CRIMINAL and self.answer.criminal == PLAYER_ONE:
            raise ValueError(f"{self.id}: criminal == 1 but Player 1 is not the Criminal")
        if len(self.rounds) != self.config.num_rounds:
            raise ValueError(f"{self.id}: expected {self.config.num_rounds} rounds")
        for t, rnd in enumerate(self.rounds, start=1):
            speakers = [s.speaker for s in rnd]
            if speakers != list(self.config.players):
                raise ValueError(f"{self.id}: round {t} must have one statement per player in order")
            for s in rnd:
                if not 1 <= s.target <= n:
                    raise ValueError(f"{self.id}: round {t} target out of range")
        if self.roles:
            for p, r in self.roles.items():
                if not 1 <= int(p) <= n or r not in ROLES:
                    raise ValueError(f"{self.id}: bad roles entry {p}: {r}")
            if self.roles.get(PLAYER_ONE, self.answer.player1_role) != self.answer.player1_role:
                raise ValueError(f"{self.id}: roles[1] disagrees with answer")
            if self.roles.get(self.answer.criminal, CRIMINAL) != CRIMINAL:
                raise ValueError(f"{self.id}: roles[criminal] is not Criminal")
            if self.has_complete_roles():
                counts = {r: 0 for r in ROLES}
                for r in self.roles.values():
                    counts[r] += 1
                if counts != self.config.role_counts:
                    raise ValueError(f"{self.id}: role counts {counts} do not match config")
                # Every Investigator must be truthful.
                for s in self.all_statements():
                    if self.roles[s.speaker] == "Investigator" and not s.holds_for(self.answer.criminal):
                        raise ValueError(f"{self.id}: Investigator {s.speaker} made a false statement")

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "task": self.task,
            "num_players": self.config.num_players,
            "num_rounds": self.config.num_rounds,
            "variant": self.config.variant,
            "role_counts": dict(self.config.role_counts),
            "roles": {str(p): r for p, r in sorted(self.roles.items())},
            "displayed_role": self.displayed_role,
            "rounds": [[s.to_dict() for s in rnd] for rnd in self.rounds],
            "answer": self.answer.to_dict(),
            "solution": self.solution,
            "reasoning": self.reasoning,
            "meta": dict(self.meta),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        cfg = GameConfig.from_dict(d)
        rounds = [[Statement.from_dict(s) for s in rnd] for rnd in d["rounds"]]
        roles = {int(p): r for p, r in (d.get("roles") or {}).items()}
        return cls(
            id=str(d["id"]),
            config=cfg,
            displayed_role=d["displayed_role"],
            rounds=rounds,
            answer=Answer.from_dict(d["answer"]),
            roles=roles,
            solution=d.get("solution"),
            reasoning=d.get("reasoning"),
            meta=dict(d.get("meta") or {}),
            task=d.get("task", TASK_NAME),
        )


def make_scenario_id(config: GameConfig, index: int, prefix: str = "hrd") -> str:
    """Canonical id, e.g. ``hrd-n6-full-00001`` (``index`` is 1-based)."""
    return f"{prefix}-n{config.num_players}-{config.variant}-{index:05d}"


__all__ = ["TASK_NAME", "Answer", "Scenario", "make_scenario_id"]
