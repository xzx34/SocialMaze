"""Natural-language reasoning chains for Hidden Role Deduction.

:func:`explain` narrates the search trace of :mod:`socialmaze.hrd.solver` as
a first-person deduction in plain English, deterministic for a given
scenario. The text is stored as ``Scenario.reasoning`` and mirrors what a
careful Player 1 would write down:

1. **Setup.** The role Player 1 was told, the role counts of the game (only
   roles that occur), and which true roles are therefore possible.
2. **One case per perspective.** How many Investigators must be among the
   other players, which players are ruled out as Investigators and why
   (quoting the solver's recorded reasons), the eligible players, and then the
   Investigator combinations: eliminated combinations are grouped by the
   statement that contradicted them and reported compactly, consistent
   combinations are narrated step by step down to the surviving Criminal
   candidates. The case ends with its verdict.
3. **Combination of the cases.**
4. **Ending.** When the instance is uniquely solvable the text ends with the
   exact Final Judgment block (:func:`socialmaze.hrd.prompts.answer_block`);
   otherwise it ends with a sentence listing the possible Criminals and roles.

The narration never names a role that does not occur in the game, so it can
be shown to a model that was told the role counts without leaking anything.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .prompts import answer_block
from .rules import (
    CRIMINAL,
    INVESTIGATOR,
    LUNATIC,
    PLAYER_ONE,
    RUMORMONGER,
    GameConfig,
    Statement,
)
from .scenario import Scenario
from .solver import Hypothesis, Perspective, Solution, Step, solve

# --------------------------------------------------------------------------
# Phrases
# --------------------------------------------------------------------------

ARTICLES = {INVESTIGATOR: "an", CRIMINAL: "the", RUMORMONGER: "a", LUNATIC: "a"}


def role_with_article(role: str) -> str:
    """``"an Investigator"``, ``"the Criminal"``, ``"a Rumormonger"``, ``"a Lunatic"``."""
    return f"{ARTICLES[role]} {role}"


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def join_words(items: Sequence[str]) -> str:
    """``"a"``, ``"a and b"``, ``"a, b and c"``."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def players_phrase(players: Iterable[int]) -> str:
    """``"Player 4"`` or ``"Players 2, 3 and 6"``."""
    players = sorted(players)
    if len(players) == 1:
        return f"Player {players[0]}"
    return "Players " + join_words([str(p) for p in players])


def one_of(players: Iterable[int]) -> str:
    """``"Player 4"``, ``"Player 1 (me)"``, or ``"one of Players 4 and 5"``."""
    players = sorted(players)
    if players == [PLAYER_ONE]:
        return "Player 1 (me)"
    if len(players) == 1:
        return f"Player {players[0]}"
    return f"one of {players_phrase(players)}"


def combo_phrase(investigators: Sequence[int]) -> str:
    return "{" + ", ".join(str(p) for p in investigators) + "}"


def describe(statement: Statement) -> str:
    """``"Player 2 said that Player 6 is not the criminal"`` with I/me for Player 1."""
    who = "I" if statement.speaker == PLAYER_ONE else f"Player {statement.speaker}"
    verb = "is" if statement.accuses else "is not"
    if statement.target == PLAYER_ONE:
        what = "I am the criminal" if statement.accuses else "I am not the criminal"
    else:
        what = f"Player {statement.target} {verb} the criminal"
    return f"{who} said that {what}"


def role_counts_phrase(config: GameConfig) -> str:
    """``"3 Investigators, 1 Criminal, 1 Rumormonger and 1 Lunatic"`` (present roles only)."""
    return join_words([plural(config.role_counts[r], r) for r in config.roles_present()])


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def setup_lines(config: GameConfig, displayed_role: str, perspectives: Sequence[Perspective]) -> list[str]:
    told = role_with_article(displayed_role)
    lines = [
        f"I was told that I am {told}.",
        f"This game has {config.num_players} players: {role_counts_phrase(config)}. "
        "Only Investigators always tell the truth.",
    ]
    roles = [p.p1_role for p in perspectives]
    if len(roles) == 1:
        lines.append(
            f"In this game every player who is told that they are {told} really is "
            f"{told}, so I am {role_with_article(roles[0])}."
        )
    else:
        lines.append(
            f"{roles[1]}s are told the same, so I am either {role_with_article(roles[0])} "
            f"or {role_with_article(roles[1])}, and I consider both cases."
        )
    return lines


# --------------------------------------------------------------------------
# One perspective
# --------------------------------------------------------------------------


def case_heading(index: int, total: int, role: str) -> str:
    if total == 1:
        return f"The only case is that I am {role_with_article(role)}."
    return f"Case {index}: I am {role_with_article(role)}."


def case_setup(perspective: Perspective) -> str:
    k = perspective.k
    if perspective.p1_role == INVESTIGATOR:
        if k == 0:
            others = "none of the other players is an Investigator"
        elif k == 1:
            others = "one of the other players is also an Investigator"
        else:
            others = f"{k} of the other players are also Investigators"
        return f"Then {others}, and my own statements are true."
    if k == 1:
        where = "the only Investigator is one of the other players"
    else:
        where = f"all {k} Investigators are among the other players"
    if perspective.p1_role == CRIMINAL:
        return (
            f"Then I am the Criminal and {where}; my own statements may be false, "
            "so I ignore them, and every statement of an Investigator must agree that "
            "I am the criminal."
        )
    return f"Then I am not the Criminal and {where}; my own statements may be false, so I ignore them."


def combo_label(hypothesis: Hypothesis) -> str:
    other = "other " if hypothesis.p1_role == INVESTIGATOR else ""
    if not hypothesis.investigators:
        return "If no other player is an Investigator"
    return f"If the {other}Investigators are {players_phrase(hypothesis.investigators)}"


def contradiction_phrase(step: Step) -> str:
    s = step.statement
    if s.accuses:
        target = "I am" if s.target == PLAYER_ONE else f"Player {s.target} is"
        return f"{describe(s)}, but {target} not a possible Criminal at that point"
    return f"{describe(s)}, which rules out the last possible Criminal"


def group_eliminated(hypotheses: Sequence[Hypothesis]) -> list[tuple[Step, list[Hypothesis]]]:
    """Eliminated hypotheses grouped by the statement that contradicted them, in order of appearance."""
    groups: dict[Statement, tuple[Step, list[Hypothesis]]] = {}
    for h in hypotheses:
        step = h.first_contradiction
        if step.statement not in groups:
            groups[step.statement] = (step, [])
        groups[step.statement][1].append(h)
    return list(groups.values())


def narrate_eliminated(step: Step, group: Sequence[Hypothesis]) -> str:
    reason = contradiction_phrase(step)
    if len(group) == 1:
        return f"{combo_label(group[0])}: {reason}, so this combination is impossible."
    combos = join_words([combo_phrase(h.investigators) for h in group])
    return f"The combinations {combos} are impossible for the same reason: {reason}."


def narrate_consistent(hypothesis: Hypothesis) -> str:
    own_steps = [st for st in hypothesis.steps if st.statement.speaker == PLAYER_ONE]
    other_steps = [st for st in hypothesis.steps if st.statement.speaker != PLAYER_ONE]
    intro = f"{combo_label(hypothesis)}, the Criminal must be {one_of(hypothesis.initial_candidates)}"
    if own_steps and own_steps[-1].after != hypothesis.initial_candidates:
        intro += f", and my own statements leave {one_of(own_steps[-1].after)}"
    sentences = [intro + "."]
    silent: list[int] = []
    for st in other_steps:
        if not st.changed:
            if st.statement.speaker not in silent:
                silent.append(st.statement.speaker)
            continue
        if st.statement.accuses:
            sentences.append(f"{describe(st.statement)}, so the Criminal is {one_of(st.after)}.")
        else:
            sentences.append(f"{describe(st.statement)}, which leaves {one_of(st.after)}.")
    if silent:
        which = "other statements" if len(sentences) > 1 else "statements"
        sentences.append(f"The {which} of {players_phrase(silent)} rule out nothing further.")
    sentences.append(
        f"So this combination is consistent, and the Criminal would be {one_of(hypothesis.candidates)}."
    )
    return " ".join(sentences)


def narrate_criminal_case(perspective: Perspective) -> list[str]:
    """The Criminal perspective: every remaining combination agrees that Player 1 is the Criminal."""
    count = len(perspective.hypotheses)
    return [
        f"{'Each' if count > 1 else 'This'} combination is consistent with me being the Criminal, "
        "since none of these players denies that I am the criminal or accuses anyone else.",
        "So the case where I am the Criminal is consistent with the statements.",
    ]


def perspective_lines(index: int, total: int, perspective: Perspective) -> list[str]:
    role = perspective.p1_role
    k = perspective.k
    lines = [case_heading(index, total, role), case_setup(perspective)]
    if perspective.p1_contradiction:
        lines.append(perspective.p1_contradiction)
        return lines
    lines.extend(perspective.p1_statement_notes)
    if perspective.excluded:
        lines.append("First I check who cannot be an Investigator in this case.")
        lines.extend(perspective.excluded.values())
    else:
        lines.append("No player is ruled out as an Investigator by their statements in this case.")
    eligible = perspective.eligible
    if not perspective.enough_eligible:
        if not eligible:
            who = "No player could be an Investigator"
        elif len(eligible) == 1:
            who = f"Only {players_phrase(eligible)} could be an Investigator"
        else:
            who = f"Only {players_phrase(eligible)} could be Investigators"
        lines.append(f"{who}, but {k} {'is' if k == 1 else 'are'} needed, so this case is impossible.")
        lines.append(f"So I cannot be {role_with_article(role)}.")
        return lines
    count = len(perspective.hypotheses)
    if k == 0:
        lines.append("No other player needs to be an Investigator, so there is 1 combination to check.")
    else:
        lines.append(
            f"So the possible Investigators are {players_phrase(eligible)}, and {k} of them must be "
            f"Investigators: {plural(count, 'combination')} to check."
        )
    if role == CRIMINAL:
        lines.extend(narrate_criminal_case(perspective))
        return lines
    eliminated = perspective.eliminated_hypotheses
    consistent = perspective.consistent_hypotheses
    if eliminated:
        if consistent:
            lines.append(f"Of these, {plural(len(eliminated), 'combination')} lead{'s' if len(eliminated) == 1 else ''} to a contradiction:")
        else:
            lines.append(f"{'All' if count > 1 else 'The only'} {plural(count, 'combination') if count > 1 else 'combination'} lead{'s' if count == 1 else ''} to a contradiction:")
        for step, group in group_eliminated(eliminated):
            lines.append(narrate_eliminated(step, group))
    if consistent:
        if eliminated and len(consistent) == 1:
            lines.append("The remaining combination is consistent:")
        elif eliminated:
            lines.append(f"The remaining {len(consistent)} combinations are consistent:")
        else:
            lines.append(f"{'All' if count > 1 else 'The only'} {plural(count, 'combination') if count > 1 else 'combination'} {'is' if count == 1 else 'are'} consistent:")
        for h in consistent:
            lines.append(narrate_consistent(h))
        lines.append(f"So if I am {role_with_article(role)}, the Criminal is {one_of(perspective.criminals)}.")
    else:
        lines.append(f"Every combination leads to a contradiction, so I cannot be {role_with_article(role)}.")
    return lines


# --------------------------------------------------------------------------
# Combination and ending
# --------------------------------------------------------------------------


def roles_phrase(roles: Sequence[str]) -> str:
    return join_words([role_with_article(r) for r in roles]).replace(" and ", " or ")


def conclusion_lines(solution: Solution) -> list[str]:
    perspectives = solution.perspectives
    possible = [p for p in perspectives if p.possible]
    if not possible:
        return ["No case is consistent with the statements, so the statements contradict the rules."]
    if len(perspectives) == 1:
        role = perspectives[0].p1_role
        if solution.unique:
            return [f"Therefore I am {role_with_article(role)}, and the Criminal is {one_of(solution.possible_criminals)}."]
        return [
            f"Therefore I am {role_with_article(role)}, but the Criminal could be "
            f"{one_of(solution.possible_criminals)}."
        ]
    if len(possible) == 1:
        impossible = [p.p1_role for p in perspectives if not p.possible]
        lead = (
            f"Comparing the cases: the case where I am {role_with_article(impossible[0])} is impossible, "
            f"so I am {role_with_article(possible[0].p1_role)}, "
        )
        if solution.unique:
            return [lead + f"and the Criminal is {one_of(solution.possible_criminals)}."]
        return [lead + f"but the Criminal could be {one_of(solution.possible_criminals)}."]
    same = len(solution.possible_criminals) == 1
    return [
        "Comparing the cases: both cases are consistent with the statements, so my role is not determined; "
        + (
            f"the Criminal is {one_of(solution.possible_criminals)} in both cases."
            if same
            else f"the Criminal could be {one_of(solution.possible_criminals)}."
        )
    ]


def open_ending(solution: Solution) -> str:
    criminals = solution.possible_criminals
    roles = solution.possible_p1_roles
    if not criminals or not roles:
        return "The statements do not determine an answer, because no case survives."
    criminal_part = f"the Criminal is {one_of(criminals)}" if len(criminals) == 1 else f"the Criminal could be {one_of(criminals)}"
    role_part = f"I am {role_with_article(roles[0])}" if len(roles) == 1 else f"I could be {roles_phrase(roles)}"
    return f"The statements do not determine a unique answer: {criminal_part}, and {role_part}."


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def explain(scenario: Scenario, solution: Optional[Solution] = None) -> str:
    """First-person deduction for ``scenario`` (solved on all its statements unless ``solution`` is given)."""
    if solution is None:
        solution = solve(scenario.config, scenario.displayed_role, scenario.all_statements())
    blocks: list[list[str]] = [setup_lines(solution.config, solution.displayed_role, solution.perspectives)]
    total = len(solution.perspectives)
    for i, perspective in enumerate(solution.perspectives, start=1):
        blocks.append(perspective_lines(i, total, perspective))
    blocks.append(conclusion_lines(solution))
    if solution.unique:
        blocks.append([answer_block(solution.criminal, solution.player1_role)])
    else:
        blocks.append([open_ending(solution)])
    return "\n\n".join("\n".join(lines) for lines in blocks)


__all__ = [
    "role_with_article", "players_phrase", "one_of", "describe", "role_counts_phrase",
    "explain",
]
