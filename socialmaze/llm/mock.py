"""Offline stand-in for a model, used by the tests and for trying the pipeline.

The mock never touches the network. From the ``context`` that
:mod:`socialmaze.hrd.evaluate` passes with every call (the ground-truth
``answer``, the round index, the numbers of rounds and players, the scenario
id) it writes a short fake reasoning paragraph followed by a Final Judgment
block in the exact format of :data:`socialmaze.hrd.prompts.FINAL_JUDGMENT_FORMAT`.
Each behaviour exercises one path of the evaluator:

============  ==============================================================
oracle        correct Criminal and correct role (100% on every metric)
wrong         another player and another role (0% on every metric)
unknown       correct Criminal, role "Unknown" (Crim. 100%, Self 0%)
garbage       no Final Judgment block at all (extraction failure)
truncate      reply cut before the block, ``finish_reason`` "length"
flaky         the first call for each (scenario id, round) fails with an
              error, later calls behave like ``oracle`` (resume and retry)
hedged        two roles on the role line (hedged answer, scored as wrong)
============  ==============================================================

Select a behaviour with ``--models mock`` (oracle), ``mock:<behaviour>`` or
one of the ``mock-<behaviour>`` presets in ``configs/models.yaml``. Replies are
deterministic; token counts are estimated as ``len(text) // 4``.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

from ..hrd.prompts import answer_block
from ..hrd.rules import CRIMINAL, DISPLAYED_ROLE, INVESTIGATOR, LUNATIC, ROLES, RUMORMONGER, UNKNOWN
from .client import TRUNCATED_FINISH_REASON, BaseClient, ChatResult

ORACLE = "oracle"
BEHAVIOURS: tuple[str, ...] = (ORACLE, "wrong", "unknown", "garbage", "truncate", "flaky", "hedged")
DEFAULT_BEHAVIOUR = ORACLE

#: Error string returned by the first call of the ``flaky`` behaviour.
TRANSIENT_ERROR = "MockError: transient"

#: Hedged role line per displayed role: the two roles that share the label.
HEDGES: dict[str, str] = {
    INVESTIGATOR: f"{INVESTIGATOR} or {RUMORMONGER}",
    CRIMINAL: f"{CRIMINAL} or {LUNATIC}",
}


def answer_fields(context: dict) -> tuple[int, str]:
    """``(criminal, player1_role)`` from ``context["answer"]`` (object or dict).

    Without an answer the mock cannot know the truth and answers
    ``Player 1`` / ``Unknown``.
    """
    answer = context.get("answer")
    if answer is None:
        return 1, UNKNOWN
    if isinstance(answer, dict):
        return int(answer["criminal"]), str(answer["player1_role"])
    return int(answer.criminal), str(answer.player1_role)


def wrong_criminal(criminal: int, num_players: int) -> int:
    """The next player after ``criminal`` (wrapping), never ``criminal`` itself."""
    return criminal % num_players + 1


def wrong_role(role: str) -> str:
    """The next role in canonical order, never ``role`` itself."""
    if role not in ROLES:
        return ROLES[0]
    return ROLES[(ROLES.index(role) + 1) % len(ROLES)]


def estimate_tokens(text: str) -> int:
    return len(text) // 4


class MockClient(BaseClient):
    """Deterministic offline client; see the module docstring for behaviours."""

    def __init__(self, behaviour: str = DEFAULT_BEHAVIOUR) -> None:
        if behaviour not in BEHAVIOURS:
            raise ValueError(
                f"unknown mock behaviour {behaviour!r}; expected one of {', '.join(BEHAVIOURS)}"
            )
        self.behaviour = behaviour
        self.calls = 0
        self._lock = threading.Lock()
        self._seen: set[tuple] = set()

    def chat(
        self,
        messages: Sequence[dict],
        temperature: float,
        max_tokens: Optional[int] = None,
        context: Optional[dict] = None,
    ) -> ChatResult:
        context = context or {}
        with self._lock:
            self.calls += 1
        if self.behaviour == "flaky" and self._first_call(context):
            return ChatResult(text="", error=TRANSIENT_ERROR)
        text, finish_reason = self.reply(context)
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        return ChatResult(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=estimate_tokens(" " * prompt_chars),
            completion_tokens=estimate_tokens(text),
            latency_s=0.0,
        )

    def _first_call(self, context: dict) -> bool:
        key = (context.get("scenario_id"), context.get("round"))
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            return True

    def reply(self, context: dict) -> tuple[str, str]:
        """``(text, finish_reason)`` of the reply for ``context``."""
        criminal, role = answer_fields(context)
        reasoning = self.reasoning(context)
        if self.behaviour == "garbage":
            return f"{reasoning}\n\nI cannot commit to an answer yet.", "stop"
        if self.behaviour == "truncate":
            return f"{reasoning} Looking at Player", TRUNCATED_FINISH_REASON
        if self.behaviour == "wrong":
            num_players = int(context.get("num_players") or criminal + 1)
            criminal, role = wrong_criminal(criminal, num_players), wrong_role(role)
        elif self.behaviour == "unknown":
            role = UNKNOWN
        elif self.behaviour == "hedged":
            role = HEDGES[DISPLAYED_ROLE.get(role, INVESTIGATOR)]
        return f"{reasoning}\n\n{answer_block(criminal, role)}", "stop"

    def reasoning(self, context: dict) -> str:
        """A short deterministic paragraph standing in for the model's reasoning."""
        round_index = context.get("round")
        num_rounds = context.get("num_rounds")
        where = f"round {round_index} of {num_rounds}" if round_index is not None else "all rounds"
        return (
            f"Mock reasoning ({self.behaviour}, {where}): I check every statement made so "
            "far against the role counts and keep only the role assignments in which all "
            "Investigators told the truth."
        )


__all__ = [
    "BEHAVIOURS", "DEFAULT_BEHAVIOUR", "TRANSIENT_ERROR", "HEDGES", "answer_fields",
    "wrong_criminal", "wrong_role", "estimate_tokens", "MockClient",
]
