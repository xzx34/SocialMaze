# Hidden Role Deduction: rules

Hidden Role Deduction (HRD) is the core task of SocialMaze (paper Section 3.1
and Appendix B). It is a social deduction game in which the model plays one
player, watches every player make public statements for several rounds, and
must work out two things: who the Criminal is, and what its own true role is.
The second question is what makes the task unusual: the model is told a role
at the start of the game, and that role may be wrong.

The definitions below are implemented in `socialmaze/hrd/rules.py`, which is
the single source of truth; this page restates them in prose.

## Players and roles

A game has `n` players, numbered 1 to `n`. Player 1 is the player whose
perspective the model takes. Every player has exactly one hidden role:

| Role | Told at the start | Statements | Notes |
|---|---|---|---|
| Investigator | "Investigator" | always true | the only reliable source of information |
| Criminal | "Criminal" | may be true or false | exactly one per game; knows they are the Criminal |
| Rumormonger | "Investigator" | may be true or false | believes they are an Investigator |
| Lunatic | "Criminal" | may be true or false | believes they are the Criminal, but is not |

The role a player is told is called the *displayed role*. Investigators and
Rumormongers are both told "Investigator"; the Criminal and Lunatics are both
told "Criminal". Consequently a player told "Investigator" is either an
Investigator or a Rumormonger, and a player told "Criminal" is either the
Criminal or a Lunatic, restricted to the roles that exist in the game.

The key rule: only Investigators are guaranteed to be truthful. Statements of
the Criminal, of Rumormongers and of Lunatics carry no information on their
own. A statement such as "Player 3 is not the criminal" is a constraint only
if it was made by an Investigator, and which players are Investigators is
exactly what has to be inferred.

## Rounds and statements

The game lasts `T` rounds (three by default). In every round each player,
including Player 1, makes exactly one public statement about another player,
in player order. A statement has the fixed form

```
Player <speaker> says Player <target> is the criminal.
Player <speaker> says Player <target> is not the criminal.
```

with `target != speaker`. Player 1's own statements are produced by the
simulator like everyone else's: the model is a passive observer that reasons
about the transcript, it does not choose what Player 1 says.

## Variants

The role counts define four named variants. There is always exactly one
Criminal; `x` is the number of Rumormongers and `y` the number of Lunatics, and
the remaining `n - 1 - x - y` players are Investigators.

| Variant | Rumormongers `x` | Lunatics `y` | Uncertainty for Player 1 |
|---|---|---|---|
| `original` | 0 | 0 | none: the displayed role is the true role |
| `rumormonger` | at least 1 | 0 | a player told "Investigator" may be a Rumormonger |
| `lunatic` | 0 | at least 1 | a player told "Criminal" may be a Lunatic |
| `full` | at least 1 | at least 1 | both |

The default number of each special role is `max(1, n // 5)`, which gives the
configurations of the released data:

| Players | Variant | Investigators | Criminal | Rumormongers | Lunatics |
|---|---|---|---|---|---|
| 6 | original | 5 | 1 | 0 | 0 |
| 6 | rumormonger | 4 | 1 | 1 | 0 |
| 6 | lunatic | 4 | 1 | 0 | 1 |
| 6 | full | 3 | 1 | 1 | 1 |
| 10 | original | 9 | 1 | 0 | 0 |
| 10 | rumormonger | 7 | 1 | 2 | 0 |
| 10 | lunatic | 7 | 1 | 0 | 2 |
| 10 | full | 5 | 1 | 2 | 2 |

The paper's experiments use the six-player `full` variant. The HuggingFace
release calls the six-player full variant `easy` and the ten-player full
variant `hard`. The original code called the full variant `all`; that name is
still accepted as an alias.

## What the model is asked

The model receives a system prompt describing the setup, the role behaviours,
the key rule, the role it was told, the task and the required output format
(`socialmaze/hrd/prompts.py`). In the evaluation protocol of the paper
(*incremental* mode) the statements of each round are then sent as a new user
message, the model's previous replies stay in the conversation, and after every
round the model must end its reply with

```
Final Judgment:
Final Criminal Is Player [number].
My Role Is [Investigator/Criminal/Rumormonger/Lunatic/Unknown].
```

This yields a Round 1 / Round 2 / Round 3 accuracy curve. In *final* mode all
rounds are shown at once and the model answers once. The model may answer
"Unknown" for its own role; that answer is scored as incorrect but is
distinguished from a malformed reply in the error analysis.

## A worked example

Scenario `hrd-n6-full-00006` from `data/hrd/hrd_n6_full.jsonl`: six players,
full variant, Player 1 was told "Investigator". The transcript is what the
model sees; the roles in brackets are hidden.

```
Round 1:
  Player 1 says Player 6 is the criminal.          [Rumormonger, false]
  Player 2 says Player 4 is the criminal.          [Investigator, true]
  Player 3 says Player 6 is not the criminal.      [Investigator, true]
  Player 4 says Player 6 is the criminal.          [Criminal, false]
  Player 5 says Player 3 is the criminal.          [Lunatic, false]
  Player 6 says Player 4 is the criminal.          [Investigator, true]
Round 2:
  Player 1 says Player 2 is the criminal.          [Rumormonger, false]
  Player 2 says Player 3 is not the criminal.      [Investigator, true]
  Player 3 says Player 4 is the criminal.          [Investigator, true]
  Player 4 says Player 2 is the criminal.          [Criminal, false]
  Player 5 says Player 4 is the criminal.          [Lunatic, true]
  Player 6 says Player 5 is not the criminal.      [Investigator, true]
Round 3:
  Player 1 says Player 5 is the criminal.          [Rumormonger, false]
  Player 2 says Player 6 is not the criminal.      [Investigator, true]
  Player 3 says Player 5 is not the criminal.      [Investigator, true]
  Player 4 says Player 1 is the criminal.          [Criminal, false]
  Player 5 says Player 1 is the criminal.          [Lunatic, false]
  Player 6 says Player 2 is not the criminal.      [Investigator, true]
```

The reasoning chain stored with the scenario (`socialmaze-hrd solve
data/hrd/hrd_n6_full.jsonl --explain hrd-n6-full-00006`) reads:

```
I was told that I am an Investigator.
This game has 6 players: 3 Investigators, 1 Criminal, 1 Rumormonger and 1 Lunatic. Only Investigators always tell the truth.
Rumormongers are told the same, so I am either an Investigator or a Rumormonger, and I consider both cases.

Case 1: I am an Investigator.
Then 2 of the other players are also Investigators, and my own statements are true.
I said that Player 6 is the criminal and also that Player 2 is the criminal, which cannot both be true, so I cannot be an Investigator.

Case 2: I am a Rumormonger.
Then I am not the Criminal and all 3 Investigators are among the other players; my own statements may be false, so I ignore them.
First I check who cannot be an Investigator in this case.
Player 4 said that I am the criminal, but I am not the Criminal in this case, so Player 4 cannot be an Investigator.
Player 5 said that I am the criminal, but I am not the Criminal in this case, so Player 5 cannot be an Investigator.
So the possible Investigators are Players 2, 3 and 6, and 3 of them must be Investigators: 1 combination to check.
The only combination is consistent:
If the Investigators are Players 2, 3 and 6, the Criminal must be one of Players 4 and 5. Player 2 said that Player 4 is the criminal, so the Criminal is Player 4. The other statements of Players 2, 3 and 6 rule out nothing further. So this combination is consistent, and the Criminal would be Player 4.
So if I am a Rumormonger, the Criminal is Player 4.

Comparing the cases: the case where I am an Investigator is impossible, so I am a Rumormonger, and the Criminal is Player 4.

Final Judgment:
Final Criminal Is Player 4.
My Role Is Rumormonger.
```

Here Player 1 can discover its true role from its own contradictory
statements. In many instances the evidence is indirect instead: the
Investigator count, together with what the other players say about each
other and about Player 1, leaves only one consistent assignment. The
`solution` block records after which round that happens.

## Unique solvability

An instance is *uniquely solvable* when, over all assignments of roles to
players that respect the role counts, the role Player 1 was told and the
truthfulness of every Investigator statement, both the Criminal and Player 1's
true role are the same in every consistent assignment. Only uniquely solvable
instances are kept in the datasets, so every instance has a single correct
answer that follows from the transcript by logic alone. The solver in
`socialmaze/hrd/solver.py` decides this exhaustively; `docs/hrd/data.md`
describes the algorithm and how it relates to Algorithm 1 in the paper.

Two consequences are worth keeping in mind when reading results:

* Whether Player 1 can determine its own role depends on the evidence the
  other players happen to produce. A Rumormonger told "Investigator" can only
  discover the truth if a real Investigator's statement contradicts one of its
  own, or if the Investigator count forces it. The `solution` block of every
  scenario records after which round the instance became solvable and whether
  an Investigator ever made a statement about Player 1.
* The two questions are scored separately (`Crim.` and `Self`) and jointly
  (`Both`). See `docs/hrd/evaluation.md`.
