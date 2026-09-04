"""Dataset generation for Hidden Role Deduction.

:func:`generate_dataset` produces uniquely solvable games by rejection
sampling: it simulates a game with :func:`socialmaze.hrd.simulate.simulate_game`,
runs the exhaustive solver (:func:`socialmaze.hrd.solver.analyze`) and keeps
the game only if both the Criminal and Player 1's true role are determined by
the statements. Accepted scenarios receive sequential ids, the solver's
verdict in ``solution``, the narrated deduction of
:func:`socialmaze.hrd.explain.explain` in ``reasoning`` and provenance in
``meta``.

Player 1's true role is controlled by a *role mix* (:func:`parse_role_mix`):

* ``"uniform"`` (default): the same number of accepted scenarios for every
  role that occurs in the game, so a dataset tests every perspective equally.
  This differs from the original release, whose Player 1 was almost always a
  Rumormonger or a Lunatic.
* ``"natural"``: no control; Player 1's role has the frequency it would have
  under a uniformly random role assignment (proportional to the role counts).
* explicit weights such as ``"Investigator=1,Criminal=1,Rumormonger=2,Lunatic=2"``.

Quotas per role are apportioned with the largest-remainder method so that
they add up exactly to the requested number of scenarios. Everything is
driven by one ``random.Random(seed)``, so a (config, seed, role mix, policy)
tuple always yields the same dataset. :func:`run_generate` implements the
``generate`` command of the CLI and writes the JSONL file plus a
``.meta.json`` sidecar with the settings and acceptance statistics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
import shlex
from collections import Counter
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from .. import __version__
from .explain import explain
from .io import save_scenarios
from .rules import ROLES, GameConfig
from .scenario import TASK_NAME, Scenario, make_scenario_id
from .simulate import Policy, make_policy, simulate_game
from .solver import analyze

#: Normalised weights per role present in the game, or ``None`` for "natural".
RoleMix = Optional[dict[str, float]]

# --------------------------------------------------------------------------
# Role mix
# --------------------------------------------------------------------------


def match_role(name: str) -> str:
    key = name.strip().lower()
    for role in ROLES:
        if role.lower() == key:
            return role
    raise ValueError(f"unknown role {name!r}; expected one of {', '.join(ROLES)}")


def parse_role_mix(spec: str, config: GameConfig) -> RoleMix:
    """Parse a ``--role-mix`` value into normalised weights over the roles present.

    ``"uniform"`` gives equal weight to every role of the game, ``"natural"``
    returns ``None`` (no control over Player 1's role), and an explicit
    ``Role=weight,...`` list is normalised. A role that does not occur in the
    game may be omitted or given weight 0; any other weight is an error.
    """
    present = config.roles_present()
    key = spec.strip().lower()
    if key == "uniform":
        return {r: 1.0 / len(present) for r in present}
    if key == "natural":
        return None
    weights: dict[str, float] = {}
    for part in spec.split(","):
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"bad role mix entry {part.strip()!r}; expected Role=weight")
        role = match_role(name)
        try:
            weight = float(value)
        except ValueError as exc:
            raise ValueError(f"bad weight {value.strip()!r} for {role}") from exc
        if weight < 0:
            raise ValueError(f"negative weight for {role}")
        if role not in present:
            if weight != 0:
                raise ValueError(
                    f"{role} does not occur in the {config.variant!r} variant with "
                    f"{config.num_players} players; give it weight 0 or omit it"
                )
            continue
        weights[role] = weights.get(role, 0.0) + weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("the role mix must give positive weight to at least one role of the game")
    return {r: weights[r] / total for r in present if weights.get(r, 0.0) > 0}


def allocate_quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` scenarios over the weighted roles."""
    raw = {r: w * total for r, w in weights.items()}
    quotas = {r: math.floor(x) for r, x in raw.items()}
    leftover = total - sum(quotas.values())
    by_remainder = sorted(weights, key=lambda r: (-(raw[r] - quotas[r]), ROLES.index(r)))
    for r in by_remainder[:leftover]:
        quotas[r] += 1
    return quotas


def choose_p1_role(
    config: GameConfig,
    weights: RoleMix,
    quotas: Optional[dict[str, int]],
    accepted: Counter,
    rng: random.Random,
) -> str:
    """Player 1's role for the next simulated game."""
    if weights is None:
        roles = list(config.roles_present())
        return rng.choices(roles, weights=[config.role_counts[r] for r in roles], k=1)[0]
    open_roles = [r for r in weights if accepted[r] < quotas[r]]
    return rng.choices(open_roles, weights=[weights[r] for r in open_roles], k=1)[0]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generation_stats(
    config: GameConfig,
    scenarios: list[Scenario],
    attempts: int,
    attempts_by_role: Counter,
    accepted_by_role: Counter,
) -> dict:
    """Acceptance and solvability statistics of one generation run."""
    per_role = {}
    for role in config.roles_present():
        a, c = attempts_by_role[role], accepted_by_role[role]
        per_role[role] = {"attempts": a, "accepted": c, "acceptance_rate": c / a if a else None}
    after = Counter(str(s.solution["solvable_after_round"]) for s in scenarios)
    cross = [s.solution["p1_cross_checked_by_investigator"] for s in scenarios]
    return {
        "attempts": attempts,
        "accepted": len(scenarios),
        "acceptance_rate": len(scenarios) / attempts if attempts else None,
        "per_role": per_role,
        "solvable_after_round": {k: after[k] for k in sorted(after, key=int)},
        "cross_checked_rate": sum(cross) / len(cross) if cross else None,
        "p1_role_counts": {r: accepted_by_role[r] for r in config.roles_present()},
    }


def generate_dataset(
    config: GameConfig,
    num_scenarios: int,
    role_mix: Union[str, RoleMix] = "uniform",
    targeting: Union[str, Policy] = "random",
    seed: int = 0,
    max_attempts_factor: int = 100,
    progress: bool = False,
    id_prefix: str = "hrd",
) -> tuple[list[Scenario], dict]:
    """Generate ``num_scenarios`` uniquely solvable scenarios and their statistics.

    Raises ``RuntimeError`` when more than ``num_scenarios * max_attempts_factor``
    games had to be simulated, which only happens for configurations in which
    unique solvability is very rare.
    """
    if num_scenarios < 0:
        raise ValueError("num_scenarios must be non-negative")
    rng = random.Random(seed)
    policy = make_policy(targeting) if isinstance(targeting, str) else targeting
    weights = parse_role_mix(role_mix, config) if isinstance(role_mix, str) else role_mix
    quotas = allocate_quotas(weights, num_scenarios) if weights is not None else None
    attempts_by_role: Counter = Counter()
    accepted_by_role: Counter = Counter()
    scenarios: list[Scenario] = []
    attempts = 0
    max_attempts = num_scenarios * max_attempts_factor
    bar = tqdm(
        total=num_scenarios,
        disable=not progress,
        desc=f"hrd n{config.num_players} {config.variant}",
        unit="scenario",
    )
    while len(scenarios) < num_scenarios:
        if attempts >= max_attempts:
            bar.close()
            raise RuntimeError(
                f"only {len(scenarios)} of {num_scenarios} scenarios were uniquely solvable "
                f"after {attempts} simulated games; raise max_attempts_factor"
            )
        attempts += 1
        p1_role = choose_p1_role(config, weights, quotas, accepted_by_role, rng)
        attempts_by_role[p1_role] += 1
        scenario = simulate_game(config, p1_role, policy, rng)
        solution = analyze(scenario)
        if not solution["unique"]:
            continue
        accepted_by_role[p1_role] += 1
        scenario.id = make_scenario_id(config, len(scenarios) + 1, id_prefix)
        scenario.solution = solution
        scenario.reasoning = explain(scenario)
        scenario.meta = {
            "generator_version": __version__,
            "targeting": policy.name,
            "seed": seed,
            "attempt": attempts,
        }
        scenario.validate()
        scenarios.append(scenario)
        bar.update(1)
        bar.set_postfix(attempts=attempts, refresh=False)
    bar.close()
    return scenarios, generation_stats(config, scenarios, attempts, attempts_by_role, accepted_by_role)


# --------------------------------------------------------------------------
# The ``generate`` command
# --------------------------------------------------------------------------


def default_output_path(config: GameConfig) -> Path:
    return Path("data") / "hrd" / f"hrd_n{config.num_players}_{config.variant}.jsonl"


def reconstruct_command(args: argparse.Namespace, config: GameConfig, out: Path) -> str:
    """The canonical command line that reproduces this run."""
    parts = ["python", "-m", "socialmaze.hrd", "generate",
             "-n", str(config.num_players), "--variant", config.variant]
    if args.rumormongers is not None:
        parts += ["--rumormongers", str(args.rumormongers)]
    if args.lunatics is not None:
        parts += ["--lunatics", str(args.lunatics)]
    if config.num_rounds != 3:
        parts += ["--num-rounds", str(config.num_rounds)]
    parts += ["-N", str(args.num_scenarios), "--seed", str(args.seed),
              "--targeting", args.targeting, "--role-mix", args.role_mix]
    if args.max_attempts_factor != 100:
        parts += ["--max-attempts-factor", str(args.max_attempts_factor)]
    parts += ["--out", str(out)]
    return " ".join(shlex.quote(p) for p in parts)


def format_rate(count: int, total: int) -> str:
    return f"{count}/{total}" + (f" ({100.0 * count / total:.1f}%)" if total else "")


def format_summary(out: Path, stats: dict) -> str:
    lines = [
        f"wrote {stats['accepted']} scenarios to {out}",
        f"accepted {format_rate(stats['accepted'], stats['attempts'])} simulated games",
    ]
    for role, c in stats["per_role"].items():
        lines.append(f"  Player 1 is {role}: {format_rate(c['accepted'], c['attempts'])}")
    hist = ", ".join(f"round {k}: {v}" for k, v in stats["solvable_after_round"].items())
    lines.append(f"first uniquely solvable after: {hist}")
    return "\n".join(lines)


def run_generate(args: argparse.Namespace) -> int:
    """The ``generate`` CLI command."""
    try:
        config = GameConfig.create(
            args.num_players, args.variant, args.rumormongers, args.lunatics, args.num_rounds
        )
        parse_role_mix(args.role_mix, config)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    out = Path(args.out) if args.out is not None else default_output_path(config)
    if out.exists() and not args.overwrite:
        print(f"error: {out} exists; pass --overwrite to replace it")
        return 1
    scenarios, stats = generate_dataset(
        config,
        args.num_scenarios,
        role_mix=args.role_mix,
        targeting=args.targeting,
        seed=args.seed,
        max_attempts_factor=args.max_attempts_factor,
        progress=not args.quiet,
    )
    meta = {
        "task": TASK_NAME,
        "generator_version": __version__,
        "created": dt.date.today().isoformat(),
        "config": config.to_dict(),
        "num_scenarios": len(scenarios),
        "seed": args.seed,
        "targeting": args.targeting,
        "role_mix": args.role_mix,
        "stats": stats,
        "command": reconstruct_command(args, config, out),
    }
    save_scenarios(out, scenarios, meta)
    print(format_summary(out, stats))
    return 0


__all__ = [
    "RoleMix", "match_role", "parse_role_mix", "allocate_quotas", "choose_p1_role",
    "generation_stats", "generate_dataset", "default_output_path",
    "reconstruct_command", "format_summary", "run_generate",
]
