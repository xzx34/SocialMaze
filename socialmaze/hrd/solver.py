"""Exhaustive uniqueness solver for Hidden Role Deduction.

The solver decides, from Player 1's point of view, which players can be the
Criminal and which true role Player 1 can have given the public statements.
An instance is *uniquely solvable* when both are determined. The search
follows Algorithm 1 of the paper and is exhaustive: its verdict is exactly
the verdict of enumerating every assignment of roles to players that agrees
with the role counts, with the role Player 1 was told, and with the rule that
Investigators always tell the truth (``tests/brute_force.py`` does that
enumeration and the test-suite checks the two agree).

Search structure
----------------
1. **Perspectives.** Player 1's true role is one of
   ``config.true_roles_for(displayed_role)``: Investigator or Rumormonger when
   told "Investigator", Criminal or Lunatic when told "Criminal", restricted to
   roles that actually occur in the game. A perspective fixes whether Player 1
   is the Criminal, how many Investigators ``k`` are among players 2..n, and
   whether Player 1's own statements are true (only as an Investigator).
2. **Quick exclusions.** A player whose statements can never all be true in
   this perspective is not an Investigator: they accused Player 1 although
   Player 1 is not the Criminal here, cleared Player 1 although Player 1 is
   the Criminal here, accused someone else although Player 1 is the Criminal
   here, accused two different players, or both accused and cleared the same
   player. In the Investigator perspective a player accused by Player 1 is
   excluded as well, because Player 1 is truthful there, and conflicting
   claims by Player 1 themselves are recorded as ruling the perspective out
   (the enumeration below reaches the same verdict on its own). Each exclusion is
   recorded with its reason so the explanation can quote it. These rules are
   sound, and the enumeration below does not rely on them being complete.
3. **Hypotheses.** For every ``k``-subset ``S`` of the remaining players the
   solver assumes ``S`` are exactly the Investigators among players 2..n and
   propagates their statements (plus Player 1's, in the Investigator
   perspective) over the set of Criminal candidates: the players outside
   ``S`` and Player 1, or just Player 1 when Player 1 is the Criminal. "Player
   v is the criminal" narrows the candidates to ``{v}`` or fails when ``v`` is
   not a candidate; "Player v is not the criminal" removes ``v``. The
   hypothesis is consistent iff candidates remain. Since the roles that are
   not Investigator or Criminal impose no constraint, every consistent
   ``(perspective, S, candidate)`` triple corresponds to at least one full
   role assignment, and vice versa, which is what makes the search exact.
4. **Verdict.** The possible Criminals are the union of the surviving
   candidates over *all* consistent hypotheses of all perspectives, including
   hypotheses that leave several candidates; the possible roles are the
   perspectives with at least one consistent hypothesis. The instance is
   unique iff both sets are singletons.

Every hypothesis keeps its propagation trace (:class:`Step`) so that
:mod:`socialmaze.hrd.explain` can narrate the deduction. :func:`analyze`
packages the verdict of a scenario, per round prefix, into the dictionary
stored in ``Scenario.solution``; :func:`run_solve` implements the ``solve``
command of the CLI.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional, Sequence

from .rules import CRIMINAL, INVESTIGATOR, PLAYER_ONE, GameConfig, Statement
from .scenario import Scenario

# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One truthful statement applied to the Criminal candidates."""

    statement: Statement
    before: frozenset[int]
    after: frozenset[int]
    contradiction: bool

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass(frozen=True)
class Hypothesis:
    """Player 1 has ``p1_role`` and ``investigators`` are the Investigators among players 2..n."""

    p1_role: str
    investigators: tuple[int, ...]
    steps: tuple[Step, ...]
    consistent: bool
    candidates: frozenset[int]

    @property
    def truthful(self) -> tuple[int, ...]:
        """Players whose statements are assumed true under this hypothesis."""
        if self.p1_role == INVESTIGATOR:
            return (PLAYER_ONE,) + self.investigators
        return self.investigators

    @property
    def initial_candidates(self) -> frozenset[int]:
        """Criminal candidates before any statement was applied."""
        return self.steps[0].before if self.steps else self.candidates

    @property
    def first_contradiction(self) -> Optional[Step]:
        """The step that emptied the candidates, if the hypothesis failed."""
        return self.steps[-1] if self.steps and self.steps[-1].contradiction else None


@dataclass
class Perspective:
    """Everything the solver derived under one assumption about Player 1's true role."""

    p1_role: str
    k: int
    excluded: dict[int, str]
    eligible: list[int]
    p1_statement_notes: list[str]
    hypotheses: list[Hypothesis]
    possible: bool
    criminals: set[int]
    p1_contradiction: Optional[str] = None

    @property
    def consistent_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.consistent]

    @property
    def eliminated_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if not h.consistent]

    @property
    def enough_eligible(self) -> bool:
        return len(self.eligible) >= self.k

    def to_dict(self) -> dict:
        return {
            "p1_role": self.p1_role,
            "k": self.k,
            "excluded": {str(p): reason for p, reason in self.excluded.items()},
            "eligible": list(self.eligible),
            "num_hypotheses": len(self.hypotheses),
            "num_consistent": len(self.consistent_hypotheses),
            "possible": self.possible,
            "criminals": sorted(self.criminals),
            "p1_contradiction": self.p1_contradiction,
        }


@dataclass
class Solution:
    """The solver's verdict on one set of statements, with the full search trace."""

    config: GameConfig
    displayed_role: str
    perspectives: list[Perspective]
    possible_criminals: list[int]
    possible_p1_roles: list[str]
    unique: bool
    criminal: Optional[int]
    player1_role: Optional[str]
    num_consistent_hypotheses: int

    def to_dict(self) -> dict:
        return {
            "displayed_role": self.displayed_role,
            "config": self.config.to_dict(),
            "unique": self.unique,
            "criminal": self.criminal,
            "player1_role": self.player1_role,
            "possible_criminals": list(self.possible_criminals),
            "possible_p1_roles": list(self.possible_p1_roles),
            "num_consistent_hypotheses": self.num_consistent_hypotheses,
            "perspectives": [p.to_dict() for p in self.perspectives],
        }


# --------------------------------------------------------------------------
# Quick exclusions
# --------------------------------------------------------------------------


def conflicting_claims(own: Sequence[Statement]) -> Optional[str]:
    """``"said that ... and also that ..., which cannot both be true"`` if a speaker's claims conflict."""
    accused: list[int] = []
    for s in own:
        if s.accuses and s.target not in accused:
            accused.append(s.target)
    if len(accused) > 1:
        return (
            f"said that Player {accused[0]} is the criminal and also that "
            f"Player {accused[1]} is the criminal, which cannot both be true"
        )
    cleared = {s.target for s in own if not s.accuses}
    for v in accused:
        if v in cleared:
            return (
                f"said that Player {v} is the criminal and also that "
                f"Player {v} is not the criminal, which cannot both be true"
            )
    return None


def exclusion_reason(
    player: int, statements: Sequence[Statement], p1_role: str
) -> Optional[str]:
    """Why ``player`` (2..n) cannot be an Investigator in the ``p1_role`` perspective, or ``None``."""
    p1_is_criminal = p1_role == CRIMINAL
    if p1_role == INVESTIGATOR:
        for s in statements:
            if s.speaker == PLAYER_ONE and s.target == player and s.accuses:
                return (
                    f"Player {player} is the Criminal by my own statement, "
                    f"so Player {player} cannot be an Investigator."
                )
    own = [s for s in statements if s.speaker == player]
    for s in own:
        if s.target == PLAYER_ONE and s.accuses and not p1_is_criminal:
            return (
                f"Player {player} said that I am the criminal, but I am not the Criminal "
                f"in this case, so Player {player} cannot be an Investigator."
            )
        if s.target == PLAYER_ONE and not s.accuses and p1_is_criminal:
            return (
                f"Player {player} said that I am not the criminal, but I am the Criminal "
                f"in this case, so Player {player} cannot be an Investigator."
            )
        if s.target != PLAYER_ONE and s.accuses and p1_is_criminal:
            return (
                f"Player {player} said that Player {s.target} is the criminal, but I am "
                f"the Criminal in this case, so Player {player} cannot be an Investigator."
            )
    conflict = conflicting_claims(own)
    if conflict is not None:
        return f"Player {player} {conflict}, so Player {player} cannot be an Investigator."
    return None


def p1_contradiction(statements: Sequence[Statement], p1_role: str) -> Optional[str]:
    """Why Player 1 cannot be an Investigator because of their own statements, or ``None``."""
    if p1_role != INVESTIGATOR:
        return None
    conflict = conflicting_claims([s for s in statements if s.speaker == PLAYER_ONE])
    if conflict is None:
        return None
    return f"I {conflict}, so I cannot be an Investigator."


def p1_statement_notes(statements: Sequence[Statement], p1_role: str) -> list[str]:
    """What Player 1's own statements establish when Player 1 is an Investigator."""
    if p1_role != INVESTIGATOR:
        return []
    notes: list[str] = []
    for s in statements:
        if s.speaker != PLAYER_ONE:
            continue
        if s.accuses:
            note = f"I said that Player {s.target} is the criminal, so Player {s.target} is the Criminal."
        else:
            note = f"I said that Player {s.target} is not the criminal, so Player {s.target} is not the Criminal."
        if note not in notes:
            notes.append(note)
    return notes


# --------------------------------------------------------------------------
# Constraint propagation
# --------------------------------------------------------------------------


def apply_statement(candidates: frozenset[int], statement: Statement) -> frozenset[int]:
    """Candidates that remain when ``statement`` is true."""
    if statement.accuses:
        return frozenset({statement.target}) if statement.target in candidates else frozenset()
    return candidates - {statement.target}


def propagate(
    config: GameConfig,
    p1_role: str,
    investigators: Sequence[int],
    statements: Sequence[Statement],
) -> Hypothesis:
    """Assume ``investigators`` are the Investigators among players 2..n and propagate."""
    investigators = tuple(sorted(investigators))
    if p1_role == CRIMINAL:
        candidates = frozenset({PLAYER_ONE})
    else:
        candidates = frozenset(config.players) - {PLAYER_ONE} - set(investigators)
    truthful: list[Statement] = []
    if p1_role == INVESTIGATOR:
        truthful.extend(s for s in statements if s.speaker == PLAYER_ONE)
    truthful.extend(s for s in statements if s.speaker in investigators)
    steps: list[Step] = []
    for s in truthful:
        after = apply_statement(candidates, s)
        steps.append(Step(s, candidates, after, not after))
        candidates = after
        if not candidates:
            break
    return Hypothesis(p1_role, investigators, tuple(steps), bool(candidates), candidates)


def solve_perspective(
    config: GameConfig, p1_role: str, statements: Sequence[Statement]
) -> Perspective:
    """Run exclusions and the hypothesis enumeration for one assumed role of Player 1."""
    others = [p for p in config.players if p != PLAYER_ONE]
    k = config.num_investigators - (1 if p1_role == INVESTIGATOR else 0)
    excluded: dict[int, str] = {}
    for u in others:
        reason = exclusion_reason(u, statements, p1_role)
        if reason is not None:
            excluded[u] = reason
    eligible = [u for u in others if u not in excluded]
    hypotheses = [propagate(config, p1_role, combo, statements) for combo in combinations(eligible, k)]
    criminals: set[int] = set()
    for h in hypotheses:
        if h.consistent:
            criminals.update(h.candidates)
    return Perspective(
        p1_role=p1_role,
        k=k,
        excluded=excluded,
        eligible=eligible,
        p1_statement_notes=p1_statement_notes(statements, p1_role),
        hypotheses=hypotheses,
        possible=bool(criminals),
        criminals=criminals,
        p1_contradiction=p1_contradiction(statements, p1_role),
    )


def check_statements(config: GameConfig, statements: Sequence[Statement]) -> None:
    n = config.num_players
    for s in statements:
        if not (1 <= s.speaker <= n and 1 <= s.target <= n):
            raise ValueError(f"statement {s} refers to a player outside 1..{n}")


def solve(
    config: GameConfig, displayed_role: str, statements: Iterable[Statement]
) -> Solution:
    """Decide the possible Criminals and Player 1 roles given ``statements``."""
    statements = list(statements)
    check_statements(config, statements)
    perspectives = [
        solve_perspective(config, role, statements)
        for role in config.true_roles_for(displayed_role)
    ]
    criminals: set[int] = set()
    for p in perspectives:
        criminals.update(p.criminals)
    possible_criminals = sorted(criminals)
    possible_roles = [p.p1_role for p in perspectives if p.possible]
    unique = len(possible_criminals) == 1 and len(possible_roles) == 1
    return Solution(
        config=config,
        displayed_role=displayed_role,
        perspectives=perspectives,
        possible_criminals=possible_criminals,
        possible_p1_roles=possible_roles,
        unique=unique,
        criminal=possible_criminals[0] if unique else None,
        player1_role=possible_roles[0] if unique else None,
        num_consistent_hypotheses=sum(len(p.consistent_hypotheses) for p in perspectives),
    )


def solve_prefixes(
    config: GameConfig, displayed_role: str, rounds: Sequence[Sequence[Statement]]
) -> list[Solution]:
    """One :class:`Solution` per round prefix ``1..T`` (what Player 1 knows after each round)."""
    solutions: list[Solution] = []
    seen: list[Statement] = []
    for rnd in rounds:
        seen.extend(rnd)
        solutions.append(solve(config, displayed_role, seen))
    return solutions


# --------------------------------------------------------------------------
# Scenario analysis
# --------------------------------------------------------------------------


def cross_checked_by_investigator(scenario: Scenario) -> Optional[bool]:
    """Whether a true Investigator made a statement about Player 1 (``None`` if roles are unknown)."""
    if not scenario.has_complete_roles():
        return None
    return any(
        s.target == PLAYER_ONE and scenario.roles[s.speaker] == INVESTIGATOR
        for s in scenario.all_statements()
    )


def analyze(scenario: Scenario) -> dict:
    """The solver's verdict on a scenario, in the form stored in ``Scenario.solution``."""
    solutions = solve_prefixes(scenario.config, scenario.displayed_role, scenario.rounds)
    final = solutions[-1] if solutions else solve(scenario.config, scenario.displayed_role, [])
    solvable_after_round = next((t for t, s in enumerate(solutions, start=1) if s.unique), None)
    return {
        "unique": final.unique,
        "criminal": final.criminal,
        "player1_role": final.player1_role,
        "possible_criminals": list(final.possible_criminals),
        "possible_p1_roles": list(final.possible_p1_roles),
        "num_consistent_hypotheses": final.num_consistent_hypotheses,
        "solvable_after_round": solvable_after_round,
        "possible_criminals_by_round": [list(s.possible_criminals) for s in solutions],
        "p1_cross_checked_by_investigator": cross_checked_by_investigator(scenario),
    }


# --------------------------------------------------------------------------
# The ``solve`` command
# --------------------------------------------------------------------------


def answer_agrees(scenario: Scenario, analysis: dict) -> bool:
    """Whether the stored answer is among the solver's possible answers."""
    return (
        scenario.criminal in analysis["possible_criminals"]
        and scenario.player1_role in analysis["possible_p1_roles"]
    )


def solve_statistics(scenarios: Sequence[Scenario]) -> dict:
    """Uniqueness statistics and agreement checks over a dataset."""
    by_role: dict[str, Counter] = {}
    after_round: Counter = Counter()
    cross = Counter()
    unique = 0
    answer_disagreements: list[str] = []
    solution_disagreements: list[str] = []
    without_solution = 0
    for sc in scenarios:
        analysis = analyze(sc)
        unique += analysis["unique"]
        role_counter = by_role.setdefault(sc.player1_role, Counter())
        role_counter["scenarios"] += 1
        role_counter["unique"] += analysis["unique"]
        after_round[str(analysis["solvable_after_round"] or "never")] += 1
        checked = analysis["p1_cross_checked_by_investigator"]
        if checked is not None:
            cross["known"] += 1
            cross["checked"] += checked
        if not answer_agrees(sc, analysis):
            answer_disagreements.append(sc.id)
        if sc.solution is None:
            without_solution += 1
        elif sc.solution != analysis:
            solution_disagreements.append(sc.id)
    total = len(scenarios)
    return {
        "scenarios": total,
        "unique": unique,
        "unique_rate": unique / total if total else None,
        "unique_by_p1_role": {
            role: {
                "scenarios": c["scenarios"],
                "unique": c["unique"],
                "unique_rate": c["unique"] / c["scenarios"],
            }
            for role, c in sorted(by_role.items())
        },
        "solvable_after_round": dict(sorted(after_round.items())),
        "cross_checked": cross["checked"],
        "cross_checked_rate": cross["checked"] / cross["known"] if cross["known"] else None,
        "answer_disagreements": answer_disagreements,
        "without_stored_solution": without_solution,
        "solution_disagreements": solution_disagreements,
    }


def format_rate(count: int, total: int) -> str:
    return f"{count}/{total}" + (f" ({100.0 * count / total:.1f}%)" if total else "")


def format_statistics(stats: dict) -> str:
    total = stats["scenarios"]
    lines = [
        f"scenarios: {total}",
        f"uniquely solvable: {format_rate(stats['unique'], total)}",
    ]
    for role, c in stats["unique_by_p1_role"].items():
        lines.append(f"  Player 1 is {role}: {format_rate(c['unique'], c['scenarios'])}")
    hist = ", ".join(f"round {k}: {v}" if k != "never" else f"never: {v}"
                     for k, v in stats["solvable_after_round"].items())
    lines.append(f"first uniquely solvable after: {hist}")
    if stats["cross_checked_rate"] is None:
        lines.append("Player 1 cross-checked by a true Investigator: unknown (roles not stored)")
    else:
        lines.append(
            "Player 1 cross-checked by a true Investigator: "
            f"{100.0 * stats['cross_checked_rate']:.1f}%"
        )
    bad = stats["answer_disagreements"]
    lines.append(f"stored answers consistent with the solver: {format_rate(total - len(bad), total)}")
    if bad:
        lines.append("  disagreeing: " + ", ".join(bad))
    with_solution = total - stats["without_stored_solution"]
    bad_solution = stats["solution_disagreements"]
    lines.append(
        f"stored solutions equal to the recomputed ones: "
        f"{format_rate(with_solution - len(bad_solution), with_solution)}"
        f" ({stats['without_stored_solution']} without a stored solution)"
    )
    if bad_solution:
        lines.append("  differing: " + ", ".join(bad_solution))
    return "\n".join(lines)


def run_solve(args: argparse.Namespace) -> int:
    """The ``solve`` CLI command: statistics over a dataset, or ``--explain`` one scenario."""
    from .explain import explain
    from .io import load_scenarios

    scenarios = load_scenarios(args.data, limit=args.limit)
    if args.explain is not None or args.index is not None:
        if args.index is not None:
            if not 0 <= args.index < len(scenarios):
                print(f"index {args.index} out of range (dataset has {len(scenarios)} scenarios)")
                return 1
            scenario = scenarios[args.index]
        else:
            matches = [s for s in scenarios if s.id == args.explain]
            if not matches:
                print(f"no scenario with id {args.explain!r}")
                return 1
            scenario = matches[0]
        print(explain(scenario))
        return 0
    stats = solve_statistics(scenarios)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_statistics(stats))
    return 1 if stats["answer_disagreements"] else 0


__all__ = [
    "Step", "Hypothesis", "Perspective", "Solution", "conflicting_claims",
    "exclusion_reason", "p1_contradiction", "p1_statement_notes",
    "apply_statement", "propagate", "solve_perspective",
    "solve", "solve_prefixes", "cross_checked_by_investigator", "analyze",
    "answer_agrees", "solve_statistics", "format_statistics", "run_solve",
]
