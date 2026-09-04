"""Game definition for Hidden Role Deduction (HRD).

This module is the single source of truth for the rules of the game: the four
roles, what each role is told about itself, which roles are guaranteed to be
truthful, the named variants, the default role counts, and the format of a
public statement. Every other module in :mod:`socialmaze.hrd` builds on the
definitions here, and ``docs/hrd/rules.md`` describes the same rules in prose.

Rules in brief
--------------
* ``n`` players, numbered ``1..n``. Player 1 is the player whose perspective
  the model takes. Exactly one player is the Criminal.
* Roles: Investigator (always truthful), Criminal (may lie), Rumormonger
  (told they are an Investigator, statements unreliable), Lunatic (told they
  are the Criminal but is not, statements unreliable).
* Displayed role: Investigators and Rumormongers are told "Investigator";
  Criminals and Lunatics are told "Criminal".
* The game lasts ``num_rounds`` rounds (3 by default). In each round every
  player makes exactly one public statement about another player, claiming
  that the player "is" or "is not" the criminal.
* Only Investigators are guaranteed to be truthful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------

INVESTIGATOR = "Investigator"
CRIMINAL = "Criminal"
RUMORMONGER = "Rumormonger"
LUNATIC = "Lunatic"
UNKNOWN = "Unknown"

#: All roles, in the canonical order used for role counts and reports.
ROLES: tuple[str, ...] = (INVESTIGATOR, CRIMINAL, RUMORMONGER, LUNATIC)

#: What each role is told about itself at the start of the game.
DISPLAYED_ROLE: dict[str, str] = {
    INVESTIGATOR: INVESTIGATOR,
    CRIMINAL: CRIMINAL,
    RUMORMONGER: INVESTIGATOR,
    LUNATIC: CRIMINAL,
}

#: The two labels a player can be shown.
DISPLAYED_ROLES: tuple[str, ...] = (INVESTIGATOR, CRIMINAL)

#: Roles whose statements are guaranteed to be true.
TRUTHFUL_ROLES: frozenset[str] = frozenset({INVESTIGATOR})

#: One-sentence behaviour description per role, reused by prompts and docs.
ROLE_BEHAVIOR: dict[str, str] = {
    INVESTIGATOR: (
        "Investigators always tell the truth: an Investigator says that a "
        "player is the criminal only if that player really is the Criminal, "
        "and says that a player is not the criminal only if that player "
        "really is not."
    ),
    CRIMINAL: (
        "The Criminal knows that they are the Criminal and may say true or "
        "false things about other players to avoid being identified."
    ),
    RUMORMONGER: (
        "Rumormongers are told that they are Investigators and believe it, "
        "but their statements are unreliable: each statement may be true or "
        "false."
    ),
    LUNATIC: (
        "Lunatics are told that they are the Criminal and believe it, but "
        "they are not the Criminal; their statements may be true or false."
    ),
}

#: The key rule, reused by prompts and docs.
KEY_RULE = (
    "Only Investigators are guaranteed to be truthful. Because Rumormongers "
    "are told that they are Investigators and Lunatics are told that they are "
    "the Criminal, the role a player was told may not be their true role."
)

PLAYER_ONE = 1

# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

ORIGINAL = "original"
RUMORMONGER_VARIANT = "rumormonger"
LUNATIC_VARIANT = "lunatic"
FULL = "full"

#: Named variants, matching the paper (Section 3.1 and Appendix B).
VARIANTS: tuple[str, ...] = (ORIGINAL, RUMORMONGER_VARIANT, LUNATIC_VARIANT, FULL)

#: Accepted spellings for variants (the legacy code called ``full`` ``all``).
VARIANT_ALIASES: dict[str, str] = {"all": FULL}


def normalize_variant(name: str) -> str:
    """Return the canonical variant name for ``name`` (case-insensitive)."""
    key = name.strip().lower()
    key = VARIANT_ALIASES.get(key, key)
    if key not in VARIANTS:
        raise ValueError(
            f"unknown variant {name!r}; expected one of {', '.join(VARIANTS)}"
        )
    return key


def variant_from_counts(num_rumormongers: int, num_lunatics: int) -> str:
    """Infer the variant name from the number of Rumormongers and Lunatics."""
    if num_rumormongers == 0 and num_lunatics == 0:
        return ORIGINAL
    if num_lunatics == 0:
        return RUMORMONGER_VARIANT
    if num_rumormongers == 0:
        return LUNATIC_VARIANT
    return FULL


def default_special_counts(num_players: int, variant: str) -> tuple[int, int]:
    """Default ``(num_rumormongers, num_lunatics)`` for a variant.

    The default number of each special role is ``max(1, num_players // 5)``,
    which reproduces the released configurations: one Rumormonger and one
    Lunatic for six players, two of each for ten players.
    """
    variant = normalize_variant(variant)
    k = max(1, num_players // 5)
    if variant == ORIGINAL:
        return 0, 0
    if variant == RUMORMONGER_VARIANT:
        return k, 0
    if variant == LUNATIC_VARIANT:
        return 0, k
    return k, k


# --------------------------------------------------------------------------
# Game configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    """Number of players, role counts and number of rounds of one game.

    There is always exactly one Criminal; the number of Investigators is
    ``num_players - 1 - num_rumormongers - num_lunatics``.
    """

    num_players: int
    num_rumormongers: int = 0
    num_lunatics: int = 0
    num_rounds: int = 3

    def __post_init__(self) -> None:
        if self.num_players < 3:
            raise ValueError("num_players must be at least 3")
        if self.num_rumormongers < 0 or self.num_lunatics < 0:
            raise ValueError("role counts must be non-negative")
        if self.num_investigators < 1:
            raise ValueError(
                "there must be at least one Investigator: "
                f"{self.num_players} players cannot hold 1 Criminal, "
                f"{self.num_rumormongers} Rumormongers and "
                f"{self.num_lunatics} Lunatics"
            )
        if self.num_rounds < 1:
            raise ValueError("num_rounds must be at least 1")

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        num_players: int,
        variant: Optional[str] = None,
        num_rumormongers: Optional[int] = None,
        num_lunatics: Optional[int] = None,
        num_rounds: int = 3,
    ) -> "GameConfig":
        """Build a configuration from a variant name and/or explicit counts.

        * Only ``variant`` given: use the default counts of that variant.
        * Only counts given: the variant is inferred from the counts.
        * Both given: the counts must be consistent with the variant.
        * Neither given: the ``full`` variant with default counts.
        """
        if variant is not None:
            variant = normalize_variant(variant)
        if num_rumormongers is None and num_lunatics is None:
            x, y = default_special_counts(num_players, variant or FULL)
        else:
            x = int(num_rumormongers or 0)
            y = int(num_lunatics or 0)
            if variant is None:
                variant = variant_from_counts(x, y)
        cfg = cls(num_players=num_players, num_rumormongers=x,
                  num_lunatics=y, num_rounds=num_rounds)
        if variant is not None and cfg.variant != variant:
            raise ValueError(
                f"{x} Rumormongers and {y} Lunatics describe the "
                f"{cfg.variant!r} variant, not {variant!r}"
            )
        return cfg

    @classmethod
    def from_role_counts(
        cls, role_counts: dict[str, int], num_rounds: int = 3
    ) -> "GameConfig":
        """Build a configuration from a ``{role: count}`` mapping."""
        counts = {role: int(role_counts.get(role, 0)) for role in ROLES}
        if counts[CRIMINAL] != 1:
            raise ValueError("there must be exactly one Criminal")
        n = sum(counts.values())
        return cls(num_players=n, num_rumormongers=counts[RUMORMONGER],
                   num_lunatics=counts[LUNATIC], num_rounds=num_rounds)

    # -- derived quantities ------------------------------------------------

    @property
    def num_criminals(self) -> int:
        return 1

    @property
    def num_investigators(self) -> int:
        return self.num_players - 1 - self.num_rumormongers - self.num_lunatics

    @property
    def variant(self) -> str:
        return variant_from_counts(self.num_rumormongers, self.num_lunatics)

    @property
    def role_counts(self) -> dict[str, int]:
        """``{role: count}`` in canonical role order (zero counts included)."""
        return {
            INVESTIGATOR: self.num_investigators,
            CRIMINAL: self.num_criminals,
            RUMORMONGER: self.num_rumormongers,
            LUNATIC: self.num_lunatics,
        }

    @property
    def players(self) -> tuple[int, ...]:
        """Player numbers ``1..n``."""
        return tuple(range(1, self.num_players + 1))

    def roles_present(self) -> tuple[str, ...]:
        """Roles with a non-zero count, in canonical order."""
        return tuple(r for r, c in self.role_counts.items() if c > 0)

    def true_roles_for(self, displayed_role: str) -> tuple[str, ...]:
        """True roles (present in this game) that are shown ``displayed_role``.

        A player told "Investigator" is an Investigator or, if the game has
        Rumormongers, a Rumormonger; a player told "Criminal" is the Criminal
        or, if the game has Lunatics, a Lunatic.
        """
        if displayed_role not in DISPLAYED_ROLES:
            raise ValueError(f"unknown displayed role {displayed_role!r}")
        return tuple(
            r for r in self.roles_present() if DISPLAYED_ROLE[r] == displayed_role
        )

    def role_list(self) -> list[str]:
        """One entry per player, in canonical role order (unshuffled)."""
        out: list[str] = []
        for role in ROLES:
            out.extend([role] * self.role_counts[role])
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "num_players": self.num_players,
            "num_rounds": self.num_rounds,
            "variant": self.variant,
            "role_counts": dict(self.role_counts),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        if "role_counts" in d:
            cfg = cls.from_role_counts(d["role_counts"], d.get("num_rounds", 3))
        else:
            cfg = cls.create(
                d["num_players"],
                d.get("variant"),
                d.get("num_rumormongers"),
                d.get("num_lunatics"),
                d.get("num_rounds", 3),
            )
        if "num_players" in d and cfg.num_players != int(d["num_players"]):
            raise ValueError("num_players does not match role_counts")
        if "variant" in d and d["variant"] is not None:
            if normalize_variant(d["variant"]) != cfg.variant:
                raise ValueError("variant does not match role_counts")
        return cfg


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------

CLAIM_IS = "is"
CLAIM_IS_NOT = "is_not"
CLAIMS: tuple[str, ...] = (CLAIM_IS, CLAIM_IS_NOT)

_CLAIM_ALIASES = {
    "is": CLAIM_IS,
    "is the criminal": CLAIM_IS,
    "is_not": CLAIM_IS_NOT,
    "is not": CLAIM_IS_NOT,
    "isnot": CLAIM_IS_NOT,
    "is not the criminal": CLAIM_IS_NOT,
}

STATEMENT_RE = re.compile(
    r"Player\s*#?\s*(\d+)\s+says\s+(?:that\s+)?Player\s*#?\s*(\d+)\s+"
    r"(is\s+not|is)\s+the\s+criminal",
    re.IGNORECASE,
)


def normalize_claim(claim: str) -> str:
    """Map ``"is"``/``"is not"``/``"is_not"`` spellings to a canonical claim."""
    key = re.sub(r"\s+", " ", str(claim).strip().lower())
    if key not in _CLAIM_ALIASES:
        raise ValueError(f"unknown claim {claim!r}; expected 'is' or 'is_not'")
    return _CLAIM_ALIASES[key]


@dataclass(frozen=True, order=True)
class Statement:
    """One public statement: ``speaker`` claims ``target`` is / is not the criminal."""

    speaker: int
    target: int
    claim: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "speaker", int(self.speaker))
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(self, "claim", normalize_claim(self.claim))
        if self.speaker < 1 or self.target < 1:
            raise ValueError("player numbers start at 1")
        if self.speaker == self.target:
            raise ValueError("a player cannot make a statement about themselves")

    @property
    def accuses(self) -> bool:
        """True if the statement says the target *is* the criminal."""
        return self.claim == CLAIM_IS

    def holds_for(self, criminal: int) -> bool:
        """Whether the statement is true when ``criminal`` is the Criminal."""
        return (self.target == criminal) == self.accuses

    def render(self) -> str:
        """The canonical text form, e.g. ``Player 2 says Player 5 is not the criminal.``"""
        verb = "is" if self.accuses else "is not"
        return f"Player {self.speaker} says Player {self.target} {verb} the criminal."

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "target": self.target, "claim": self.claim}

    @classmethod
    def from_dict(cls, d: dict) -> "Statement":
        return cls(int(d["speaker"]), int(d["target"]), d["claim"])

    @classmethod
    def parse(cls, text: str) -> Optional["Statement"]:
        """Parse the first statement found in ``text``, or ``None``."""
        m = STATEMENT_RE.search(text)
        if not m:
            return None
        return cls(int(m.group(1)), int(m.group(2)), m.group(3))

    @classmethod
    def parse_all(cls, text: str) -> list["Statement"]:
        """Parse every statement in ``text`` in order of appearance."""
        return [cls(int(a), int(b), c) for a, b, c in STATEMENT_RE.findall(text)]


def render_statements(statements: Iterable[Statement]) -> str:
    """Render statements one per line."""
    return "\n".join(s.render() for s in statements)


__all__ = [
    "INVESTIGATOR", "CRIMINAL", "RUMORMONGER", "LUNATIC", "UNKNOWN", "ROLES",
    "DISPLAYED_ROLE", "DISPLAYED_ROLES", "TRUTHFUL_ROLES", "ROLE_BEHAVIOR",
    "KEY_RULE", "PLAYER_ONE",
    "ORIGINAL", "RUMORMONGER_VARIANT", "LUNATIC_VARIANT", "FULL", "VARIANTS",
    "VARIANT_ALIASES", "normalize_variant", "variant_from_counts",
    "default_special_counts", "GameConfig",
    "CLAIM_IS", "CLAIM_IS_NOT", "CLAIMS", "STATEMENT_RE", "normalize_claim",
    "Statement", "render_statements",
]
