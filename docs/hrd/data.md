# Hidden Role Deduction: data

This page describes the dataset files, how instances are generated, how unique
solvability is verified, and how the files relate to the original release on
the HuggingFace Hub.

## Files

Datasets are JSONL files with one game per line. The sample files in
`data/hrd/` cover every variant; larger files are produced with
`socialmaze-hrd generate` (see below). Each file may have a sidecar
`<name>.meta.json` recording the generator version, configuration, seed,
statement policy, role mix, acceptance statistics and the exact command that
produced it.

```
data/hrd/hrd_n6_full.jsonl          six players, full variant (the paper's setting)
data/hrd/hrd_n10_full.jsonl         ten players, full variant
data/hrd/hrd_n6_original.jsonl      six players, no Rumormonger, no Lunatic
data/hrd/hrd_n6_rumormonger.jsonl   six players, one Rumormonger
data/hrd/hrd_n6_lunatic.jsonl       six players, one Lunatic
```

## Record schema

```json
{
  "id": "hrd-n6-full-00001",
  "task": "Hidden Role Deduction",
  "num_players": 6,
  "num_rounds": 3,
  "variant": "full",
  "role_counts": {"Investigator": 3, "Criminal": 1, "Rumormonger": 1, "Lunatic": 1},
  "roles": {"1": "Rumormonger", "2": "Investigator", "3": "Criminal", "4": "Investigator", "5": "Lunatic", "6": "Investigator"},
  "displayed_role": "Investigator",
  "rounds": [
    [{"speaker": 1, "target": 4, "claim": "is_not"}, {"speaker": 2, "target": 6, "claim": "is_not"}, "..."],
    "...", "..."
  ],
  "answer": {"criminal": 3, "player1_role": "Rumormonger"},
  "solution": {
    "unique": true,
    "criminal": 3,
    "player1_role": "Rumormonger",
    "possible_criminals": [3],
    "possible_p1_roles": ["Rumormonger"],
    "num_consistent_hypotheses": 1,
    "solvable_after_round": 2,
    "possible_criminals_by_round": [[3, 5], [3], [3]],
    "p1_cross_checked_by_investigator": true
  },
  "reasoning": "I was told that I am an Investigator. ...\nFinal Judgment:\nFinal Criminal Is Player 3.\nMy Role Is Rumormonger.",
  "meta": {"generator_version": "1.0.0", "targeting": "random", "seed": 0, "attempt": 17}
}
```

| Field | Meaning |
|---|---|
| `roles` | the hidden true role of every player (complete for generated data; only Player 1 and the Criminal for rows imported from the HuggingFace release) |
| `displayed_role` | what Player 1 was told: `Investigator` or `Criminal` |
| `rounds` | one list per round, one statement per player in player order; `claim` is `is` or `is_not` |
| `answer` | the ground truth the model is scored against |
| `solution.unique` | whether the answer is the only one consistent with the transcript (always true in released files) |
| `solution.solvable_after_round` | the first round after which the answer is already unique (1 to `num_rounds`) |
| `solution.possible_criminals_by_round` | criminal candidates that survive after each round prefix |
| `solution.p1_cross_checked_by_investigator` | whether a real Investigator made a statement about Player 1 (the evidence-sufficiency split used in the paper's per-role analysis) |
| `reasoning` | a deterministic natural-language derivation of the answer, ending with the Final Judgment block (used as supervision for fine-tuning) |

Statements are stored structurally; the text shown to a model is rendered by
`socialmaze/hrd/prompts.py` (`Player 2 says Player 6 is not the criminal.`).

## Generation

`socialmaze-hrd generate` (implemented in `socialmaze/hrd/generate.py`) works
as follows for every accepted instance:

1. **Configuration.** Number of players `n`, variant (or explicit Rumormonger
   and Lunatic counts) and number of rounds `T`, see `docs/hrd/rules.md`.
2. **Player 1's role.** Chosen according to `--role-mix`. The default
   `uniform` gives every role that exists in the game the same share of
   instances (1:1:1:1 for the full variant), which is the mix used for the
   paper's headline numbers. `natural` keeps whatever a uniformly random
   assignment gives Player 1 (proportional to the role counts), and explicit
   weights such as `Investigator=1,Criminal=1,Rumormonger=2,Lunatic=2` are
   also accepted. Quotas are filled by rejection sampling per role.
3. **Role assignment.** The remaining roles are shuffled over players 2 to `n`.
4. **Statements.** For each of the `T` rounds every player in order picks a
   target and a claim according to the statement policy (`--targeting`,
   implemented in `socialmaze/hrd/simulate.py`):
   * `random` (default): the target is uniform over the other players;
     Investigators state the truth; the Criminal, Rumormongers and Lunatics
     say "is the criminal" with probability one half regardless of the truth.
     This reproduces the distribution of the original release.
   * `strategic`: an implementation of the heuristic strategy functions
     described in Appendix B.2 of the paper. Investigators and Rumormongers
     prefer targets that were accused by others or contradicted themselves
     and avoid targets that were cleared; the Criminal and Lunatics prefer to
     target players who accused them and mostly accuse them back. The original
     release did not ship this policy, so this is an interpretation of the
     paper's text, not the code that produced the paper's data.
5. **Verification.** The solver (next section) checks that the Criminal and
   Player 1's role are uniquely determined. Instances that are not uniquely
   solvable are discarded.
6. **Reasoning chain.** `socialmaze/hrd/explain.py` renders the solver's trace
   into first-person prose that ends with the Final Judgment block.

The acceptance rate of step 5 depends strongly on Player 1's true role (an
Investigator can rarely be sure that it is not a Rumormonger unless another
Investigator makes a statement about it) and on the number of players.
Measured over several thousand simulated games with the `random` policy:

| Configuration | Overall | Investigator | Criminal | Rumormonger | Lunatic |
|---|---|---|---|---|---|
| 6 players, full | 81% | 64% | 92% | 80% | 89% |
| 10 players, full | 63% | 48% | 68% | 67% | 66% |
| 6 players, rumormonger | 90% | | | | |
| 6 players, lunatic | 99% | | | | |
| 6 players, original | 100% | | | | |

The `.meta.json` of every sample file records the rate observed for that file.
These rates are properties of the statement policy: a policy that makes players
speak about Player 1 less often lowers them, and the figure quoted in the paper
was obtained with an earlier generator configuration.

## The solver

`socialmaze/hrd/solver.py` follows the structure of Algorithm 1 in the paper
and is exhaustive: it reaches exactly the same verdict as enumerating every
assignment of roles to players (which `tests/brute_force.py` does, and the
test suite checks the two against each other on random games of every
variant, including games that are not uniquely solvable).

1. **Perspectives.** Player 1's true role is one of the roles that exist in the
   game and are displayed as the role Player 1 was told: `Investigator` or
   `Rumormonger` when told "Investigator", `Criminal` or `Lunatic` when told
   "Criminal". In a variant without Rumormongers (or Lunatics) there is only
   one perspective.
2. **Quick exclusions.** Within a perspective, a player cannot be an
   Investigator if one of its statements is false on its own: it accused
   Player 1 although Player 1 is not the Criminal in this perspective, it
   cleared Player 1 although Player 1 is the Criminal, it accused two different
   players, or it both accused and cleared the same player.
3. **Investigator subsets.** Every subset of the remaining players with the
   required number of Investigators is a hypothesis. Player 1's own statements
   are constraints only in the perspective in which Player 1 is an
   Investigator.
4. **Propagation.** Starting from all players who could be the Criminal in
   this hypothesis, the statements of the hypothesised Investigators are
   applied in order: "X is the criminal" narrows the candidates to X (or
   contradicts), "X is not the criminal" removes X. A hypothesis is consistent
   if at least one candidate survives.
5. **Verdict.** The possible Criminals are the union of the surviving
   candidates over all consistent hypotheses of all perspectives; the possible
   roles of Player 1 are the perspectives with at least one consistent
   hypothesis. The instance is uniquely solvable when both sets have exactly
   one element.

`solvable_after_round` and `possible_criminals_by_round` come from running the
same procedure on every round prefix. `socialmaze-hrd solve <file>` re-runs the
solver on any dataset and reports the uniqueness statistics and whether the
stored answers agree; `--explain <id>` prints the reasoning chain.

### What was fixed relative to the original release

The uniqueness checker of the May 2025 scripts (`archive/hidden_role_deduction/hrd_gen.py`)
differs from the procedure above in two ways, both of which are corrected here:

* It only recorded hypotheses that narrowed the Criminal down to exactly one
  player. A consistent hypothesis that left two or more candidates was ignored,
  so a small fraction of instances that are in fact ambiguous were accepted as
  unique. On freshly simulated six-player full games this affects roughly one
  instance in three hundred, on ten-player games roughly one in thirty.
* It always considered the "I might be a Rumormonger" and "I might be a
  Lunatic" perspectives, even in variants that contain no such role. Valid
  instances of the `lunatic` and `rumormonger` variants were dropped, and the
  generated reasoning chains mention roles that do not exist in the game.

The ten sample instances per configuration that shipped with the original
release are all uniquely solvable (`tests/test_legacy_data.py` checks this by
brute force). The HuggingFace release was filtered with the old checker; if you
need a guarantee for those rows, re-verify them with the solver:

```python
from socialmaze.hrd.io import load_hf
from socialmaze.hrd.solver import analyze

rows = load_hf("MBZUAI/SocialMaze", split="easy", limit=1000)
ambiguous = [s.id for s in rows if not analyze(s)["unique"]]
```

## Relation to the HuggingFace release

The dataset `MBZUAI/SocialMaze` has two splits, `easy` (six players, full
variant) and `hard` (ten players, full variant), with 100,000 rows each and the
columns `task`, `system_prompt`, `prompt`, `answer`, `reasoning_process`,
`round 1`, `round 2`, `round 3`. It was produced by the original generator, so
it uses the original prompt wording and a Player 1 role mix dominated by the
Rumormonger and Lunatic perspectives rather than the uniform mix of the paper.

* `socialmaze-hrd evaluate --from-hf --split easy` evaluates models on those
  rows directly (needs `pip install -e ".[hf]"`); the rows are converted with
  `socialmaze.hrd.io.from_hf_row`, which recovers the configuration, the
  displayed role, the statements and the answer. The full role assignment is
  not part of the release, so per-role tables for such runs use Player 1's
  role only.
* `socialmaze-hrd export <file> --out rows.jsonl` writes any dataset in the
  same row format (with the current prompt text) for sharing or uploading.
* Files of the original release (`archive/hidden_role_deduction/data/*.json`)
  are also loaded transparently by every command.
