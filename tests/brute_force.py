"""Reference solver for tests: exhaustive enumeration of role assignments.

This is deliberately the simplest possible correct implementation. It
enumerates every assignment of roles to players that is compatible with the
role counts and with what Player 1 was told, and keeps the assignments in
which every Investigator statement is true. It is used to check
:mod:`socialmaze.hrd.solver`, which reaches the same answer far faster.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from socialmaze.hrd.rules import CRIMINAL, INVESTIGATOR, PLAYER_ONE, GameConfig, Statement


def multiset_permutations(items: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Distinct permutations of a multiset, in lexicographic order."""
    counts: dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    keys = sorted(counts)
    n = len(items)
    out: list[str] = []

    def rec() -> Iterator[tuple[str, ...]]:
        if len(out) == n:
            yield tuple(out)
            return
        for k in keys:
            if counts[k] > 0:
                counts[k] -= 1
                out.append(k)
                yield from rec()
                out.pop()
                counts[k] += 1

    yield from rec()


def consistent_worlds(
    config: GameConfig, displayed_role: str, statements: Iterable[Statement]
) -> list[dict[int, str]]:
    """All full role assignments consistent with the statements.

    A world is consistent when every statement made by an Investigator in that
    world is true about the world's Criminal. Statements of other roles carry
    no constraint.
    """
    statements = list(statements)
    others = [p for p in config.players if p != PLAYER_ONE]
    worlds: list[dict[int, str]] = []
    for p1_role in config.true_roles_for(displayed_role):
        remaining = config.role_list()
        remaining.remove(p1_role)
        for perm in multiset_permutations(remaining):
            roles = {PLAYER_ONE: p1_role}
            roles.update(zip(others, perm))
            criminal = next(p for p, r in roles.items() if r == CRIMINAL)
            if all(
                s.holds_for(criminal)
                for s in statements
                if roles[s.speaker] == INVESTIGATOR
            ):
                worlds.append(roles)
    return worlds


def brute_force_solve(
    config: GameConfig, displayed_role: str, statements: Iterable[Statement]
) -> dict:
    """Possible criminals / Player 1 roles and whether the instance is uniquely solvable."""
    worlds = consistent_worlds(config, displayed_role, statements)
    criminals = sorted({next(p for p, r in w.items() if r == CRIMINAL) for w in worlds})
    p1_roles = sorted({w[PLAYER_ONE] for w in worlds})
    return {
        "possible_criminals": criminals,
        "possible_p1_roles": p1_roles,
        "unique": len(criminals) == 1 and len(p1_roles) == 1,
        "num_worlds": len(worlds),
    }
