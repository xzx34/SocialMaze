<h1 align="center">SocialMaze: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments</h1>

🌐 **Project page:** https://xzx34.github.io/socialmaze/

🏛️ **Venue:** [Findings of EMNLP 2026](https://2026.emnlp.org/)

🎤 **Workshop:** [SocialSim @ COLM 2025](https://sites.google.com/view/social-sims-with-llms/home) · Spotlight Talk

📄 **Paper (accepted version):** https://xzx34.github.io/socialmaze/paper.pdf

📄 **arXiv:** https://arxiv.org/abs/2505.23713

🤗 **Dataset (Hugging Face):** https://huggingface.co/datasets/MBZUAI/SocialMaze

## Updates & News

- [09/04/2026] 🛠️ **Repository rewritten around Hidden Role Deduction:** a clean `socialmaze` package with an exhaustive solver, dataset generation, an OpenAI-compatible evaluation harness and tests. The five other tasks are archived. See [What changed](#what-changed-in-this-version).
- [08/20/2026] 🥂 **SocialMaze has been accepted to Findings of EMNLP 2026! See you in Budapest!**
- [10/10/2025] 🎤 **SocialMaze was presented as a Spotlight Talk at SocialSim @ COLM 2025 in Montréal!**

## Introduction

SocialMaze is a benchmark for evaluating and enhancing the social reasoning capabilities of Large Language Models (LLMs) in complex, evolving social environments. The paper organizes six tasks across social reasoning games, daily-life interactions and digital community platforms along three descriptive design axes: *deep reasoning*, *dynamic interaction* and *information uncertainty*. It also studies enhancement strategies: reasoning workflows help weaker short-chain-of-thought backbones but saturate on stronger reasoners, while targeted fine-tuning substantially improves structured social-reasoning tasks.

This repository is the maintained implementation of the benchmark's core task, **Hidden Role Deduction (HRD)**: rules, data generation with a verified unique solution for every instance, natural-language reasoning chains, and a model evaluation harness that reproduces the paper's protocol.

## What changed in this version

The original May 2025 release contained one generation script and one evaluation script per task. That code was written before the paper reached its final form and is hard to use and to read. The repository has been reorganized as follows:

| Task (name in the paper) | Paper section | Where it lives now |
|---|---|---|
| Hidden Role Deduction | Sec. 3.1, App. B | `socialmaze/hrd/` (rewritten, maintained) |
| Find the Spy | Sec. 3.2, App. C | `archive/find_the_spy/` (frozen) |
| Rating Estimation from Text | Sec. 3.3, App. D | `archive/rating_estimation_from_text/` (frozen) |
| Social Graph Analysis | Sec. 3.4, App. E | `archive/social_graph_analysis/` (frozen) |
| Review Decision Prediction | Sec. 3.5, App. F | `archive/review_decision_prediction/` (frozen) |
| User Profile Inference | Sec. 3.6, App. G | `archive/user_profile_inference/` (frozen) |

**Please note that the archived scripts and the Hugging Face release are partly out of date.** Both were produced by the original generator and prompt, and they predate the final evaluation protocol of the paper and the solver fix described in [`docs/hrd/data.md`](docs/hrd/data.md). Wherever they disagree with the code in this repository, this repository is authoritative. The Hugging Face data remains a valid, uniquely solvable test set and can be evaluated directly with `--from-hf` (see below); its Player 1 role mix is dominated by the Rumormonger and Lunatic perspectives, whereas the paper's numbers use a uniform mix that the generator here produces by default.

## Installation

```bash
git clone https://github.com/xzx34/SocialMaze.git
cd SocialMaze
python -m venv .venv && source .venv/bin/activate   # Python 3.10 or newer
pip install -e ".[dev]"          # add ",hf" to also load data from the Hugging Face Hub
cp .env.example .env             # then add the API keys of the providers you use
```

`pytest` runs the offline test suite (solver against brute force, generator, parser, evaluation with a mock model).

## Quick start: Hidden Role Deduction

Every command below is available both as `socialmaze-hrd` and as `python -m socialmaze.hrd`.

```bash
# 1. Generate a dataset: six players, full variant, 500 uniquely solvable games,
#    Player 1 role mix 1:1:1:1, fixed seed. Writes the JSONL file and a .meta.json sidecar.
socialmaze-hrd generate -n 6 --variant full -N 500 --seed 0 --out data/hrd/hrd_n6_full_500.jsonl

# 2. Look at one game (roles, statements with their truth value, solver verdict, prompts)
socialmaze-hrd inspect data/hrd/hrd_n6_full.jsonl --index 0 --prompt

# 3. Print the solver's reasoning chain for one game, or re-verify a whole file
socialmaze-hrd solve data/hrd/hrd_n6_full.jsonl --explain hrd-n6-full-00001
socialmaze-hrd solve data/hrd/hrd_n6_full.jsonl

# 4. Evaluate models with the paper's protocol (one Final Judgment after each round,
#    temperature 0.7, five seeds). `mock` is an offline stand-in for trying the pipeline.
socialmaze-hrd evaluate --data data/hrd/hrd_n6_full.jsonl --models mock --out runs/smoke
socialmaze-hrd evaluate --data data/hrd/hrd_n6_full_500.jsonl --models gpt-4o-mini deepseek-r1 \
    --mode incremental --seeds 5 --workers 16 --out runs/n6-full

# 5. Aggregate a run directory into summary.json and report.md
socialmaze-hrd report runs/n6-full

# 6. Evaluate directly on the Hugging Face release (easy = 6 players, hard = 10 players)
socialmaze-hrd evaluate --from-hf --split easy --limit 500 --models gpt-4o-mini --out runs/hf-easy

# 7. Export any dataset to the row format of the Hugging Face release
socialmaze-hrd export data/hrd/hrd_n6_full.jsonl --out exports/hrd_n6_full_hf.jsonl
```

Sample datasets for all variants ship in [`data/hrd/`](data/hrd/) and can be regenerated with the commands recorded in their `.meta.json` files.

### The task in brief

`n` players each hold a hidden role: Investigators (always truthful), one Criminal (may lie), Rumormongers (told they are Investigators, unreliable) and Lunatics (told they are the Criminal, but are not). Over three rounds every player publicly claims that some other player "is" or "is not" the criminal. The model is Player 1, is told a role that may be wrong, and after each round must name the Criminal and its own true role. Every released instance has a unique answer that follows from the transcript by logic alone. Full rules: [`docs/hrd/rules.md`](docs/hrd/rules.md).

| Variant | Rumormongers | Lunatics | Uncertainty for Player 1 |
|---|---|---|---|
| `original` | 0 | 0 | none |
| `rumormonger` | 1 (n=6) / 2 (n=10) | 0 | told "Investigator" may be false |
| `lunatic` | 0 | 1 / 2 | told "Criminal" may be false |
| `full` (paper, HF `easy`/`hard`) | 1 / 2 | 1 / 2 | both |

### Metrics

`Crim.` is the accuracy of the Criminal identification, `Self` the accuracy of the model's own role (strict single-label match; "Unknown" and hedged answers count as wrong), `Both` the fraction of instances with both correct. All three are reported per round with a 95% binomial confidence half-width, per Player 1 true role, and with an error decomposition (API error, truncated output, missing Final Judgment, hedged role, reasoning error). Details: [`docs/hrd/evaluation.md`](docs/hrd/evaluation.md).

## Models and API keys

Models are described in [`configs/models.yaml`](configs/models.yaml). Every provider is accessed through the OpenAI-compatible chat API, so one client covers OpenAI, DeepSeek, DeepInfra, OpenRouter, Anthropic and Gemini (through their OpenAI-compatible endpoints) and local vLLM or Ollama servers. The twelve models evaluated in the paper are preconfigured; any other model can be named on the command line as `provider/model-id` without editing the file, for example `--models openrouter/qwen/qwen3-235b-a22b`. API keys are read from the environment or from a `.env` file (see `.env.example`).

## Reproducing the paper's HRD numbers

The paper evaluates the six-player `full` variant on 500 instances with a uniform 1:1:1:1 mix of Player 1 roles, in incremental mode, at temperature 0.7, averaged over five seeds, with output caps of 4096 tokens (8192 for long-chain-of-thought models), and reports the 95% binomial confidence half-width at n = 500. The commands in the quick start implement exactly this protocol. Two caveats: the model identifiers in `configs/models.yaml` are current aliases rather than the snapshots used in 2025, and the paper's prompt asked for hidden reasoning while this harness keeps the reasoning visible, so numbers will be close to but not identical with the tables in the paper. The fine-tuning (SFT/DPO) and workflow experiments of Section 5 are not part of this repository.

## Repository layout

```
socialmaze/hrd/      rules, scenario schema, simulator, solver, explainer, generator,
                     prompts, parser, metrics, evaluator, report, io, cli
socialmaze/llm/      OpenAI-compatible client, model registry, mock client
configs/models.yaml  providers and model presets
data/hrd/            sample datasets (+ .meta.json) for every variant
docs/hrd/            rules.md, data.md, evaluation.md
tests/               offline test suite (pytest)
archive/             the original May 2025 release, frozen (see archive/README.md)
```

## Citation

If you use the SocialMaze benchmark or its datasets in your research, we kindly ask you to cite our work:

```bibtex
@inproceedings{xu2026socialmaze,
  title={{SocialMaze}: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments},
  author={Xu, Zixiang and Wang, Yanbo and Huang, Yue and Zhuang, Haomin and Zhou, Yujun and Ye, Jiayi and Li, Sixian and Song, Zirui and Gao, Lang and Wang, Chenxi and Chen, Zhaorun and Pan, Wang and Zhao, Yue and Zhao, Jieyu and Zhang, Xiangliang and Chen, Xiuying},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  month={October},
  year={2026},
  address={Budapest, Hungary},
  publisher={Association for Computational Linguistics},
  note={To appear}
}
```
