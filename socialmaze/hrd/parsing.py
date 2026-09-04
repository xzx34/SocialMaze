"""Extraction of the Final Judgment from a model reply.

The prompt asks the model to end its reply with::

    Final Judgment:
    Final Criminal Is Player [number].
    My Role Is [Investigator/Criminal/Rumormonger/Lunatic/Unknown].

Models decorate this block in many ways (bold markdown, headings, colons,
"Player #4", lower case, an article before the role, a trailing sentence) and
often restate it while reasoning, so :func:`parse_final_judgment` takes the
LAST occurrence of each line and tolerates the decorations. Two outcomes are
kept apart from a plain wrong answer because the error analysis reports them
separately: no criminal line at all (``found=False``, an extraction failure)
and a role line naming two or more different roles such as "Investigator or
Rumormonger" (``hedged=True``, ``role=None``; scored as wrong because the task
requires committing to one role). "Unknown" is a recognised role word: it is
a valid, though incorrect, answer and is counted by the ``unknown_rate``
metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .rules import CRIMINAL, INVESTIGATOR, LUNATIC, RUMORMONGER, UNKNOWN

#: Case-insensitive prefixes of a word and the role they denote.
ROLE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("investigat", INVESTIGATOR),
    ("criminal", CRIMINAL),
    ("killer", CRIMINAL),
    ("murderer", CRIMINAL),
    ("rumo", RUMORMONGER),
    ("luna", LUNATIC),
    ("unknown", UNKNOWN),
    ("uncertain", UNKNOWN),
    ("unsure", UNKNOWN),
    ("undetermined", UNKNOWN),
)

# Markdown and quoting characters models wrap the block in.
_DECOR = "[*_`#\\[\\]\"']*"
CRIMINAL_RE = re.compile(
    r"final\s+criminal\s+is\s*[:\-]?\s*" + _DECOR + r"\s*player\s*" + _DECOR + r"\s*#?\s*(\d+)",
    re.IGNORECASE,
)
ROLE_LINE_RE = re.compile(r"my\s+role\s+is\s*[:\-]?\s*(.*)", re.IGNORECASE)
_DECOR_CHARS_RE = re.compile("[*_`#\\[\\]\"']")
_CLAUSE_END_RE = re.compile(r"[.,;:!?()\n]")
_WORD_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class Prediction:
    """What the reply claims. ``found`` is whether a criminal line exists."""

    criminal: Optional[int]
    role: Optional[str]
    found: bool
    hedged: bool


def normalize_role(word: str) -> Optional[str]:
    """Canonical role for a single word, or ``None`` if it is not a role word."""
    key = word.strip().lower()
    for prefix, role in ROLE_PREFIXES:
        if key.startswith(prefix):
            return role
    return None


def roles_in_phrase(phrase: str) -> list[str]:
    """Distinct roles named in the first clause of ``phrase``, in order.

    Decorations are dropped and the phrase is cut at the first sentence or
    clause boundary, so a clarification after a comma or in parentheses
    ("Lunatic, not the Criminal") does not count as a second role, while
    "Investigator or Rumormonger" and "Investigator/Rumormonger" do.
    """
    clause = _CLAUSE_END_RE.split(_DECOR_CHARS_RE.sub(" ", phrase), maxsplit=1)[0]
    roles: list[str] = []
    for word in _WORD_RE.findall(clause):
        role = normalize_role(word)
        if role is not None and role not in roles:
            roles.append(role)
    return roles


def _last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    last = None
    for last in pattern.finditer(text):
        pass
    return last


def parse_final_judgment(text: Optional[str]) -> Prediction:
    """Parse the last Final Judgment in ``text``; see the module docstring."""
    text = text or ""
    criminal_match = _last_match(CRIMINAL_RE, text)
    criminal = int(criminal_match.group(1)) if criminal_match else None
    role: Optional[str] = None
    hedged = False
    role_match = _last_match(ROLE_LINE_RE, text)
    if role_match:
        roles = roles_in_phrase(role_match.group(1))
        if len(roles) == 1:
            role = roles[0]
        elif len(roles) > 1:
            hedged = True
    return Prediction(criminal=criminal, role=role, found=criminal is not None, hedged=hedged)


def extract_criminal(text: Optional[str]) -> Optional[int]:
    return parse_final_judgment(text).criminal


def extract_role(text: Optional[str]) -> Optional[str]:
    return parse_final_judgment(text).role


__all__ = [
    "ROLE_PREFIXES", "CRIMINAL_RE", "ROLE_LINE_RE", "Prediction", "normalize_role",
    "roles_in_phrase", "parse_final_judgment", "extract_criminal", "extract_role",
]
