# Hidden Role Deduction sample data

Small, fully regenerable datasets produced by `python -m socialmaze.hrd generate`.
Every row is one game seen from Player 1's perspective. All rows are uniquely
solvable: both the Criminal and Player 1's true role follow from the public
statements, as verified by the exhaustive solver in `socialmaze/hrd/solver.py`
(and, in the test-suite, by brute-force enumeration of all role assignments).
Each row carries the solver's verdict in `solution` and a first-person deduction
in `reasoning` that ends with the Final Judgment block a model is asked to produce.

The row schema is documented in the module docstring of `socialmaze/hrd/scenario.py`
and in `docs/hrd/data.md`. Every `.jsonl` file has a `.meta.json` sidecar with the
game configuration, seed, role mix, targeting policy, generation statistics and the
exact command that produced it.

## Files

| File | Variant (role counts) | Players | Rounds | Rows | Player 1 true role |
|---|---|---|---|---|---|
| `hrd_n6_full.jsonl` | full (3 Investigators, 1 Criminal, 1 Rumormonger, 1 Lunatic) | 6 | 3 | 100 | 25 Investigator, 25 Criminal, 25 Rumormonger, 25 Lunatic |
| `hrd_n10_full.jsonl` | full (5 Investigators, 1 Criminal, 2 Rumormongers, 2 Lunatics) | 10 | 3 | 100 | 25 Investigator, 25 Criminal, 25 Rumormonger, 25 Lunatic |
| `hrd_n6_original.jsonl` | original (5 Investigators, 1 Criminal) | 6 | 3 | 50 | 25 Investigator, 25 Criminal |
| `hrd_n6_rumormonger.jsonl` | rumormonger (4 Investigators, 1 Criminal, 1 Rumormonger) | 6 | 3 | 50 | 17 Investigator, 17 Criminal, 16 Rumormonger |
| `hrd_n6_lunatic.jsonl` | lunatic (4 Investigators, 1 Criminal, 1 Lunatic) | 6 | 3 | 50 | 17 Investigator, 17 Criminal, 16 Lunatic |

All files were generated with seed 0, `--role-mix uniform` (the same number of rows
for every role that occurs in the variant, apportioned by largest remainder) and
`--targeting random` (the statement distribution of the original release), by
generator version 1.0.0. The six-player full variant corresponds to the paper's
"easy" setting and the ten-player full variant to "hard".

## Regenerating

Run from the repository root; the files are written to `data/hrd/` by default.
Generation is deterministic for a given configuration, seed, role mix and policy.

```
python -m socialmaze.hrd generate -n 6  --variant full        -N 100 --seed 0 --overwrite
python -m socialmaze.hrd generate -n 10 --variant full        -N 100 --seed 0 --overwrite
python -m socialmaze.hrd generate -n 6  --variant original    -N 50  --seed 0 --overwrite
python -m socialmaze.hrd generate -n 6  --variant rumormonger -N 50  --seed 0 --overwrite
python -m socialmaze.hrd generate -n 6  --variant lunatic     -N 50  --seed 0 --overwrite
```

`--role-mix natural` leaves Player 1's role to chance (proportional to the role
counts), explicit weights such as `--role-mix Investigator=1,Criminal=1,Rumormonger=2,Lunatic=2`
are also accepted, and `--targeting strategic` switches to the suspicion-driven
statement policy described in the paper's appendix (see `socialmaze/hrd/simulate.py`).
To check a file, `python -m socialmaze.hrd solve data/hrd/hrd_n6_full.jsonl` re-runs
the solver and reports uniqueness statistics; `--explain <id>` prints the deduction
of one row.

## Acceptance rates observed

Rejection sampling keeps only uniquely solvable games. The rates below are the
`stats` entries of the `.meta.json` sidecars (accepted / simulated games).

| File | Overall | Investigator | Criminal | Rumormonger | Lunatic | First solvable after round 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| `hrd_n6_full.jsonl` | 100/127 (78.7%) | 25/43 (58.1%) | 25/25 (100%) | 25/31 (80.6%) | 25/28 (89.3%) | 12 / 51 / 37 |
| `hrd_n10_full.jsonl` | 100/160 (62.5%) | 25/51 (49.0%) | 25/29 (86.2%) | 25/46 (54.3%) | 25/34 (73.5%) | 0 / 44 / 56 |
| `hrd_n6_original.jsonl` | 50/50 (100%) | 25/25 (100%) | 25/25 (100%) | n/a | n/a | 47 / 3 / 0 |
| `hrd_n6_rumormonger.jsonl` | 50/57 (87.7%) | 17/23 (73.9%) | 17/17 (100%) | 16/17 (94.1%) | n/a | 23 / 23 / 4 |
| `hrd_n6_lunatic.jsonl` | 50/51 (98.0%) | 17/17 (100%) | 17/18 (94.4%) | n/a | 16/16 (100%) | 23 / 25 / 2 |

Games in which Player 1 is an Investigator are the hardest to make uniquely
solvable, because Player 1's own truthful statements remove information that the
other Investigators would otherwise have to supply.

## Relation to the HuggingFace release

The public dataset `MBZUAI/SocialMaze` (splits `easy` = six-player full variant,
`hard` = ten-player full variant) was produced by the original generator
(`archive/hidden_role_deduction/hrd_gen.py`) with an older system prompt and a
Player 1 role mix dominated by the Rumormonger and Lunatic perspectives; the files
here use a uniform role mix and the current prompt. The row format of that release
(`task`, `system_prompt`, `prompt`, `answer`, `reasoning_process`, `round 1..3`)
can be produced from any dataset with

```
python -m socialmaze.hrd export data/hrd/hrd_n6_full.jsonl --format hf --out hrd_n6_full_hf.jsonl
```

and every command accepts such rows (as well as the legacy JSON of the original
release) as input, so the two formats are interchangeable.
