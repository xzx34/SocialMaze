"""Role assignment and statement generation for Hidden Role Deduction.

A game is simulated in two steps. :func:`assign_roles` gives Player 1 a
requested role and shuffles the remaining roles over the other players.
:func:`simulate_game` then plays ``config.num_rounds`` rounds in which every
player, in player order, makes one public statement about another player.
What a player says is decided by a *targeting policy*, an object that sees the
speaker's true role, the identity of the Criminal and every statement made so
far, and returns the next :class:`~socialmaze.hrd.rules.Statement`.

Two policies ship with the package:

* :class:`RandomPolicy` (``"random"``, the default) reproduces the
  distribution of the original release: the target is uniform over the other
  players, Investigators tell the truth about the target, and every other role
  says "is the criminal" with probability one half regardless of the truth.
* :class:`StrategicPolicy` (``"strategic"``) is an interpretation of the
  suspicion-driven behaviour described in Appendix B.2 of the paper. The
  original release contained only the random policy, so this is a best-effort
  reading of the text, not a reimplementation of released code. Investigators
  and Rumormongers weight their target by how suspicious the public record
  makes each player (accusations received, self-contradictions, minus
  clearances); Investigators then state the truth and Rumormongers a coin
  flip. The Criminal, and a Lunatic who believes they are the Criminal, target
  players who accused them and accuse those players back most of the time.

Whatever the policy, an Investigator's claim is always the truth, so every
simulated game satisfies the rules and passes ``Scenario.validate()``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from .rules import (
    CLAIM_IS,
    CLAIM_IS_NOT,
    CRIMINAL,
    DISPLAYED_ROLE,
    INVESTIGATOR,
    LUNATIC,
    PLAYER_ONE,
    RUMORMONGER,
    GameConfig,
    Statement,
)
from .scenario import Answer, Scenario

# --------------------------------------------------------------------------
# Role assignment
# --------------------------------------------------------------------------


def assign_roles(config: GameConfig, p1_role: str, rng: random.Random) -> dict[int, str]:
    """Give Player 1 ``p1_role`` and shuffle the remaining roles over players 2..n."""
    if p1_role not in config.roles_present():
        raise ValueError(
            f"{p1_role!r} is not a role of the {config.variant!r} variant "
            f"with {config.num_players} players"
        )
    remaining = config.role_list()
    remaining.remove(p1_role)
    rng.shuffle(remaining)
    roles = {PLAYER_ONE: p1_role}
    roles.update(zip(config.players[1:], remaining))
    return roles


def criminal_of(roles: dict[int, str]) -> int:
    """The player holding the Criminal role."""
    return next(p for p, r in roles.items() if r == CRIMINAL)


# --------------------------------------------------------------------------
# Targeting policies
# --------------------------------------------------------------------------


def truthful_claim(target: int, criminal: int) -> str:
    """The claim an Investigator makes about ``target``."""
    return CLAIM_IS if target == criminal else CLAIM_IS_NOT


def coin_claim(rng: random.Random, prob_is: float = 0.5) -> str:
    """``"is"`` with probability ``prob_is``, otherwise ``"is_not"``."""
    return CLAIM_IS if rng.random() < prob_is else CLAIM_IS_NOT


def other_players(roles: dict[int, str], speaker: int) -> list[int]:
    """Players the speaker may talk about, in player order."""
    return [p for p in sorted(roles) if p != speaker]


class Policy(ABC):
    """How a player chooses the target and the claim of a statement."""

    name: ClassVar[str]

    @abstractmethod
    def statement(
        self,
        speaker: int,
        roles: dict[int, str],
        criminal: int,
        history: Sequence[Statement],
        rng: random.Random,
    ) -> Statement:
        """The next statement of ``speaker``.

        ``history`` holds every statement made so far, in order: all previous
        rounds and the earlier speakers of the current round. The returned
        statement never targets the speaker.
        """


class RandomPolicy(Policy):
    """Uniform random target; only Investigators are constrained by the truth.

    This is the distribution of the original release: Criminals, Lunatics and
    Rumormongers say "is the criminal" with probability one half, independent
    of whether the target is the Criminal.
    """

    name = "random"

    def statement(self, speaker, roles, criminal, history, rng) -> Statement:
        target = rng.choice(other_players(roles, speaker))
        if roles[speaker] == INVESTIGATOR:
            claim = truthful_claim(target, criminal)
        else:
            claim = coin_claim(rng)
        return Statement(speaker, target, claim)


def accusers_of(player: int, history: Sequence[Statement]) -> set[int]:
    """Players who have said that ``player`` is the criminal."""
    return {s.speaker for s in history if s.target == player and s.accuses}


def self_contradicted(player: int, history: Sequence[Statement]) -> bool:
    """Whether ``player`` accused two different players or both accused and cleared one."""
    accused = {s.target for s in history if s.speaker == player and s.accuses}
    cleared = {s.target for s in history if s.speaker == player and not s.accuses}
    return len(accused) > 1 or bool(accused & cleared)


def suspicion_weight(player: int, history: Sequence[Statement]) -> float:
    """Investigator-side target weight ``F_I``: higher for suspicious players.

    ``1 + 2 * accusations received + 3 * [self-contradicted] - clearances
    received``, floored at 0.2 so that no player is ever unreachable.
    """
    accused_by = sum(1 for s in history if s.target == player and s.accuses)
    cleared_by = sum(1 for s in history if s.target == player and not s.accuses)
    contradicted = 1 if self_contradicted(player, history) else 0
    return max(0.2, 1.0 + 2.0 * accused_by + 3.0 * contradicted - 1.0 * cleared_by)


class StrategicPolicy(Policy):
    """Suspicion-driven targeting, an interpretation of Appendix B.2 of the paper.

    The paper describes Investigators as selecting a target through a
    suspicion function ``F_I`` that favours players who contradicted
    themselves or were accused and disfavours players cleared by others,
    Rumormongers as using the same targeting with an unreliable claim, and the
    Criminal (and a Lunatic, who believes they are the Criminal) as using
    ``F_C``, which prioritises players who accused them, to divert suspicion.
    The original release only implemented the random policy, so the weights
    below are this package's reading of that description:

    * Investigator / Rumormonger: target ``u`` with weight
      :func:`suspicion_weight`; the Investigator states the truth, the
      Rumormonger says "is" with probability one half.
    * Criminal / Lunatic: target ``u`` with weight ``1 + 3 * [u accused the
      speaker]`` and say "is" with probability 0.8 if ``u`` accused the speaker
      and 0.3 otherwise.
    """

    name = "strategic"

    def statement(self, speaker, roles, criminal, history, rng) -> Statement:
        role = roles[speaker]
        others = other_players(roles, speaker)
        if role in (INVESTIGATOR, RUMORMONGER):
            weights = [suspicion_weight(u, history) for u in others]
            target = rng.choices(others, weights=weights, k=1)[0]
            claim = truthful_claim(target, criminal) if role == INVESTIGATOR else coin_claim(rng)
            return Statement(speaker, target, claim)
        accusers = accusers_of(speaker, history)
        weights = [1.0 + 3.0 * (u in accusers) for u in others]
        target = rng.choices(others, weights=weights, k=1)[0]
        claim = coin_claim(rng, 0.8 if target in accusers else 0.3)
        return Statement(speaker, target, claim)


POLICIES: dict[str, type[Policy]] = {
    RandomPolicy.name: RandomPolicy,
    StrategicPolicy.name: StrategicPolicy,
}


def make_policy(name: str, **kwargs) -> Policy:
    """Instantiate the policy registered under ``name``."""
    key = name.strip().lower()
    if key not in POLICIES:
        raise ValueError(f"unknown targeting policy {name!r}; expected one of {', '.join(POLICIES)}")
    return POLICIES[key](**kwargs)


# --------------------------------------------------------------------------
# Game simulation
# --------------------------------------------------------------------------


def simulate_game(
    config: GameConfig,
    p1_role: str,
    policy: Policy,
    rng: random.Random,
    scenario_id: str = "tmp",
) -> Scenario:
    """Play one game and return it as a validated :class:`Scenario`.

    The returned scenario has complete ``roles``, the ground-truth ``answer``,
    no ``solution`` or ``reasoning`` yet, and ``meta = {"targeting": policy.name}``.
    """
    roles = assign_roles(config, p1_role, rng)
    criminal = criminal_of(roles)
    history: list[Statement] = []
    rounds: list[list[Statement]] = []
    for _ in range(config.num_rounds):
        current: list[Statement] = []
        for speaker in config.players:
            statement = policy.statement(speaker, roles, criminal, history, rng)
            if statement.speaker != speaker or statement.target not in config.players:
                raise ValueError(f"policy {policy.name!r} returned an invalid statement {statement}")
            current.append(statement)
            history.append(statement)
        rounds.append(current)
    scenario = Scenario(
        id=scenario_id,
        config=config,
        displayed_role=DISPLAYED_ROLE[p1_role],
        rounds=rounds,
        answer=Answer(criminal, p1_role),
        roles=roles,
        solution=None,
        reasoning=None,
        meta={"targeting": policy.name},
    )
    scenario.validate()
    return scenario


__all__ = [
    "assign_roles", "criminal_of", "truthful_claim", "coin_claim", "other_players",
    "Policy", "RandomPolicy", "StrategicPolicy", "accusers_of", "self_contradicted",
    "suspicion_weight", "POLICIES", "make_policy", "simulate_game",
]
