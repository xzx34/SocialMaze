"""Canonical prompts for evaluating a model on Hidden Role Deduction.

This module is the single source of truth for the text shown to a model. The
structure follows the prompt template in the paper's appendix (Game setup,
Role behaviors, Your role, Your task, Output format). Two evaluation modes
share the same system prompt:

* ``incremental`` (the protocol of the paper): after every round the model
  receives that round's statements as a new user message, its previous
  replies stay in the conversation, and it must give an updated Final
  Judgment. See :func:`round_message`.
* ``final``: all rounds are shown at once in a single user message and the
  model answers once. See :func:`final_message`.

The model is expected to reason step by step and to end its reply with the
block in :data:`FINAL_JUDGMENT_FORMAT`; :mod:`socialmaze.hrd.parsing`
extracts the last such block from the reply.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .rules import (
    CRIMINAL,
    INVESTIGATOR,
    KEY_RULE,
    LUNATIC,
    ROLE_BEHAVIOR,
    ROLES,
    RUMORMONGER,
    UNKNOWN,
    GameConfig,
    Statement,
    render_statements,
)

#: The exact block a model must end its answer with.
FINAL_JUDGMENT_FORMAT = (
    "Final Judgment:\n"
    "Final Criminal Is Player [number].\n"
    f"My Role Is [{INVESTIGATOR}/{CRIMINAL}/{RUMORMONGER}/{LUNATIC}/{UNKNOWN}]."
)


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def role_count_phrase(config: GameConfig) -> str:
    """``"3 Investigators, 1 Criminal, 1 Rumormonger, 1 Lunatic"`` (zeros included)."""
    return ", ".join(_plural(config.role_counts[r], r) for r in ROLES)


def told_role_phrase(displayed_role: str) -> str:
    """``"an Investigator"`` or ``"the Criminal"``."""
    return "an Investigator" if displayed_role == INVESTIGATOR else "the Criminal"


def system_prompt(config: GameConfig, displayed_role: str) -> str:
    """The system prompt for a game with ``config`` where Player 1 is told ``displayed_role``."""
    n = config.num_players
    behaviors = "\n".join(f"- {ROLE_BEHAVIOR[r]}" for r in ROLES)
    return (
        "You are an expert in social reasoning. You are Player 1 in a social "
        f"deduction game called Hidden Role Deduction with {n} players in total "
        f"(Players 1 to {n}).\n"
        "\n"
        "Game setup:\n"
        f"- Roles: {role_count_phrase(config)}. Every player has exactly one role, "
        "and true roles are hidden. Exactly one player is the Criminal.\n"
        f"- The game lasts {config.num_rounds} rounds. In each round every player, "
        "including you, makes one public statement about another player, claiming "
        "that this player \"is the criminal\" or \"is not the criminal\".\n"
        "- Your own statements were made before your reasoning starts and appear "
        "in the transcript; you do not choose them.\n"
        "\n"
        "Role behaviors:\n"
        f"{behaviors}\n"
        f"- Key rule: {KEY_RULE}\n"
        "\n"
        "Your role:\n"
        f"You are Player 1, and you have been told that you are {told_role_phrase(displayed_role)}.\n"
        "\n"
        "Your task:\n"
        "After each round, use all statements made so far to deduce (1) which "
        "player is the Criminal and (2) your own true role. Reason step by step, "
        "then commit to your most likely answer. If you cannot determine your own "
        f"role, answer \"{UNKNOWN}\" for your role.\n"
        "\n"
        "Output format:\n"
        "End your reply with exactly this block, filled in:\n"
        f"{FINAL_JUDGMENT_FORMAT}"
    )


def round_block(round_index: int, statements: Iterable[Statement]) -> str:
    """``"Round t statements:"`` followed by one statement per line."""
    return f"Round {round_index} statements:\n{render_statements(statements)}"


def round_message(round_index: int, statements: Iterable[Statement]) -> str:
    """User message for round ``round_index`` in incremental mode."""
    span = "round 1" if round_index == 1 else f"rounds 1 to {round_index}"
    return (
        f"{round_block(round_index, statements)}\n\n"
        f"Based on all statements so far ({span}), give your reasoning and then "
        "your Final Judgment."
    )


def final_message(rounds: Sequence[Sequence[Statement]]) -> str:
    """Single user message with every round, for final mode."""
    blocks = "\n\n".join(round_block(t, r) for t, r in enumerate(rounds, start=1))
    return (
        f"{blocks}\n\n"
        f"Based on all {len(rounds)} rounds of statements, give your reasoning and "
        "then your Final Judgment."
    )


def answer_block(criminal: int, role: str, header: bool = True) -> str:
    """The filled-in Final Judgment block for a ground-truth or predicted answer."""
    body = f"Final Criminal Is Player {criminal}.\nMy Role Is {role}."
    return f"Final Judgment:\n{body}" if header else body


__all__ = [
    "FINAL_JUDGMENT_FORMAT", "role_count_phrase", "told_role_phrase",
    "system_prompt", "round_block", "round_message", "final_message",
    "answer_block",
]
