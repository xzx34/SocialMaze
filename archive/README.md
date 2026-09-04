# Archived task pipelines

This directory freezes the code and sample data of the original SocialMaze
release (May 2025). It is kept for reference only: nothing in here is
maintained, tested, or used by the `socialmaze` package at the repository
root.

The scripts predate the final version of the paper, so several details differ
from what the paper describes (sampling temperature, number of seeds, output
length caps, the mix of Player 1 roles, generator model pools, and so on).
Where the paper and this code disagree, the paper is authoritative. For
Hidden Role Deduction, the rewritten package in `socialmaze/hrd/` supersedes
the legacy scripts here and fixes a bug in the legacy uniqueness checker
(see `docs/hrd/data.md`).

## Mapping to the paper

| Legacy directory | Task name in the paper | Paper section | Status |
|---|---|---|---|
| `hidden_role_deduction/` | Hidden Role Deduction | Sec. 3.1, App. B | superseded by `socialmaze/hrd/` |
| `find_the_spy/` | Find the Spy | Sec. 3.2, App. C | archived |
| `rating_estimation_from_text/` | Rating Estimation from Text | Sec. 3.3, App. D | archived |
| `social_graph_analysis/` | Social Graph Analysis | Sec. 3.4, App. E | archived |
| `review_decision_prediction/` | Review Decision Prediction | Sec. 3.5, App. F | archived |
| `user_profile_inference/` | User Profile Inference | Sec. 3.6, App. G | archived |
| `utils/` | shared multi-provider LLM helper used by the scripts above | - | archived |

## Running the legacy scripts

Each task directory contains a `*_gen.py` (data generation) and a `*_eva.py`
(evaluation) script plus a `data/` folder with a few sample instances. The
scripts import `utils.tool` through `sys.path.append("..")`, so they must be
run from inside their own directory, for example:

```bash
cd archive/hidden_role_deduction
python hrd_eva.py --models gpt-4o-mini
```

The original dependency dump can be recovered from git history with
`git show 104f83c:requirements.txt` (it is UTF-16 encoded). API keys were read
from a `.env` file in the repository root by `utils/tool.py`.
