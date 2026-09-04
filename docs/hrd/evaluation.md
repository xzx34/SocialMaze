# Hidden Role Deduction: evaluation

This page describes the evaluation protocol, the prompts, how model output is
scored, the reported metrics, the result files, and how models are configured.

## Protocol

The protocol of the paper (Section 4 and Appendix B) is:

| Setting | Value |
|---|---|
| Data | six players, `full` variant, 500 instances, uniform 1:1:1:1 mix of Player 1 roles |
| Interaction | *incremental*: statements of each round sent as a new message, one Final Judgment per round |
| Sampling | temperature 0.7, five independent seeds |
| Output cap | 4096 completion tokens; 8192 for long-chain-of-thought models |
| Metrics | `Crim.`, `Self`, `Both` per round, per Player 1 role, error decomposition |
| Uncertainty | 95% binomial confidence half-width `1.96 * sqrt(p (1 - p) / n)` |

```bash
socialmaze-hrd generate -n 6 --variant full -N 500 --seed 0 --out data/hrd/hrd_n6_full_500.jsonl
socialmaze-hrd evaluate --data data/hrd/hrd_n6_full_500.jsonl --models gpt-4o-mini deepseek-r1 \
    --mode incremental --temperature 0.7 --seeds 5 --workers 16 --out runs/n6-full
socialmaze-hrd report runs/n6-full
```

`--mode final` sends all rounds in one message and asks for a single answer;
it is cheaper and corresponds to the question-answer format of the HuggingFace
release, but it does not produce the per-round curve.

## Prompts

All text shown to a model comes from `socialmaze/hrd/prompts.py`.

* The **system prompt** has five parts, following the template in the paper's
  appendix: game setup (number of players, role counts, rounds, statement
  format), role behaviours and the key rule, the role Player 1 was told, the
  task, and the output format. Print it for any scenario with
  `socialmaze-hrd inspect <file> --prompt`.
* In **incremental mode** the user message of round `t` is the block
  `Round t statements:` followed by one statement per line and the request to
  give a Final Judgment based on rounds 1 to `t`. The model's reply is kept in
  the conversation, so the request for round 3 contains the full history.
* In **final mode** a single user message contains the blocks of all rounds.

The model is asked to reason step by step and to end its reply with

```
Final Judgment:
Final Criminal Is Player [number].
My Role Is [Investigator/Criminal/Rumormonger/Lunatic/Unknown].
```

## Parsing and scoring

`socialmaze/hrd/parsing.py` takes the **last** `Final Criminal Is Player N`
and the last `My Role Is R` in the reply, tolerating markdown emphasis, a
`#` before the number, lower case, colons and trailing punctuation. Role
words are normalised by prefix (`Investigat*`, `Criminal`/`Killer`/`Murderer`,
`Rumor*`, `Luna*`, `Unknown`/`Uncertain`/`Unsure`).

Scoring is strict:

* `criminal_correct`: the predicted number equals the Criminal.
* `role_correct`: the predicted role equals Player 1's true role. `Unknown`,
  a hedged answer such as "Investigator or Rumormonger", an unrecognised word
  and a missing role line all count as wrong.
* `both_correct`: both of the above.

Every round entry also records `truncated` (the provider reported
`finish_reason == "length"`), `extraction_failed` (no Final Judgment block was
found), `hedged`, and `error` (the request failed after retries). These flags
feed the error decomposition below.

## Metrics and report

`socialmaze-hrd report <run_dir>` (or the automatic report at the end of
`evaluate`) writes `summary.json` and `report.md` with:

1. **Per model and round**: `Crim.`, `Self` and `Both` in percent with the
   95% binomial half-width, the number of scored instances `n`, and the rate
   of `Unknown` self-role answers.
2. **Per model and Player 1 true role** at the final round: `Self` and `Both`.
   This is the table that separates the Investigator and Criminal
   perspectives (where the displayed role is true) from the Rumormonger and
   Lunatic perspectives (where it is false).
3. **Error decomposition** at the final round, in mutually exclusive
   categories: API error, correct (both correct, even if the reply was cut by
   the cap), truncated output, missing Final Judgment, hedged role, reasoning
   error (well-formed but wrong), applied in that order. The categories sum
   to 100% and the correct share equals `Both` at the final round.
4. **Cost**: mean prompt and completion tokens and mean latency per call.

When several seeds are used, the accuracy is the mean over all instance and
seed pairs, the half-width is computed at that total `n`, and the standard
deviation across seeds is reported alongside.

## Result files

`evaluate` writes one directory per run:

```
runs/<name>/run.json          data source, settings, model list, scenario ids, package version
runs/<name>/<model>.jsonl     one record per (scenario, seed) for that model
runs/<name>/summary.json      aggregated metrics (written by report)
runs/<name>/report.md         the tables above (written by report)
```

Each record has the form

```json
{
  "id": "hrd-n6-full-00001", "seed": 0, "model": "gpt-4o-mini", "mode": "incremental",
  "num_players": 6, "num_rounds": 3, "variant": "full",
  "displayed_role": "Investigator", "player1_role": "Rumormonger",
  "answer": {"criminal": 3, "player1_role": "Rumormonger"},
  "rounds": [
    {"round": 1, "response": "...", "reasoning_text": null, "finish_reason": "stop",
     "prompt_tokens": 812, "completion_tokens": 431, "latency_s": 4.2,
     "pred_criminal": 5, "pred_role": "Investigator", "found": true, "hedged": false,
     "criminal_correct": false, "role_correct": false, "both_correct": false,
     "truncated": false, "extraction_failed": false, "error": null},
    "..."
  ],
  "created": "2026-09-04T10:15:00+00:00"
}
```

`reasoning_text` holds the separate reasoning stream returned by some
providers (for example DeepSeek-R1's `reasoning_content`); it is stored but
not parsed.

Runs are resumable: re-running `evaluate` with the same `--out` skips
(scenario, seed) pairs that already have a record and, by default, re-queries
pairs whose record contains an API error (`--no-retry-errors` keeps them).
Requests are sent in parallel per model (`--workers`), with exponential
backoff on rate limits and transient server errors.

## Models

`configs/models.yaml` lists providers (base URL and the environment variable
holding the key) and model presets (provider, model id, whether the model
accepts a temperature, the name of the token-cap parameter and its default
value). All providers are used through the OpenAI-compatible chat completions
API, which covers OpenAI, DeepSeek, DeepInfra, OpenRouter, the
OpenAI-compatible endpoints of Anthropic and Gemini, and local servers such as
vLLM or Ollama.

* The twelve models of the paper are preconfigured under their paper names
  (`gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, `deepseek-v3`, `deepseek-r1`,
  `qwq-32b`, `qwen-2.5-72b`, `llama-3.3-70b`, `llama-3.1-8b`, `phi-4`,
  `gemini-2.5-pro`). Their ids are current aliases, not the exact snapshots
  used in 2025.
* Any model can be addressed as `provider/model-id` on the command line, for
  example `--models deepinfra/Qwen/Qwen3-235B-A22B`, without editing the file.
* `mock`, `mock-wrong`, `mock-garbage`, `mock-truncate` and `mock-flaky` are
  offline stand-ins used by the tests; `--models mock` exercises the whole
  pipeline without an API key.
* Keys are read from the environment or from a `.env` file in the working
  directory or the repository root (`.env.example` lists the variable names).
